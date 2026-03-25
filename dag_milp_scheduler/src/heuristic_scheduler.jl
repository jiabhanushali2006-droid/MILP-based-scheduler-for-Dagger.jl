module HeuristicSchedulers

using Random
using Statistics

export greedy_scheduler,
       random_scheduler,
       minmin_scheduler,
       maxmin_scheduler,
       round_robin_scheduler,
       heft_scheduler,
       topo_sort


# ============================================================
# UTILITY: Build predecessor map
# ============================================================

function get_predecessors(nodes, edges)
    preds = Dict(i => Int[] for i in nodes)
    for (u, v) in edges
        push!(preds[v], u)
    end
    return preds
end


# ============================================================
# UTILITY: Topological sort (Kahn's algorithm)
# ============================================================

function topo_sort(nodes, edges)
    in_degree = Dict(i => 0 for i in nodes)
    for (_, v) in edges
        in_degree[v] += 1
    end

    queue = [i for i in nodes if in_degree[i] == 0]
    order = Int[]

    while !isempty(queue)
        n = popfirst!(queue)
        push!(order, n)
        for (u, v) in edges
            if u == n
                in_degree[v] -= 1
                if in_degree[v] == 0
                    push!(queue, v)
                end
            end
        end
    end

    if length(order) != length(nodes)
        error("Topological sort failed: graph contains a cycle.")
    end

    return order
end


# ============================================================
# UTILITY: Data-dependency ready time (no device awareness)
# ============================================================


function dep_ready_time(i, preds, assignment, start_times, runtime_matrix)
    valid_preds = [p for p in preds[i] if haskey(assignment, p)]
    isempty(valid_preds) && return 0.0
    return maximum(start_times[p] + runtime_matrix[(p, assignment[p])]
                   for p in valid_preds)
end


# ============================================================
# UTILITY: Earliest start — dependencies + device queue
# ============================================================

function earliest_start(i, r, preds, assignment, start_times,
                         runtime_matrix, device_free_at)
    return max(dep_ready_time(i, preds, assignment, start_times, runtime_matrix),
               device_free_at[r])
end


# ============================================================
# 1. RANDOM SCHEDULER
# ============================================================


function random_scheduler(nodes, edges, runtime_matrix, devices)
    assignment     = Dict{Int, Symbol}()
    start_times    = Dict{Int, Float64}()
    device_free_at = Dict(d => 0.0 for d in devices)

    order = topo_sort(nodes, edges)
    preds = get_predecessors(nodes, edges)

    for i in order
        r              = rand(devices)
        t              = earliest_start(i, r, preds, assignment, start_times,
                                        runtime_matrix, device_free_at)
        assignment[i]  = r
        start_times[i] = t
        device_free_at[r] = t + runtime_matrix[(i, r)]
    end

    makespan = maximum(start_times[i] + runtime_matrix[(i, assignment[i])]
                       for i in nodes)
    return assignment, start_times, makespan
end


# ============================================================
# 2. GREEDY LIST SCHEDULER
# ============================================================


function greedy_scheduler(nodes, edges, runtime_matrix, devices)
    assignment     = Dict{Int, Symbol}()
    start_times    = Dict{Int, Float64}()
    device_free_at = Dict(d => 0.0 for d in devices)

    order = topo_sort(nodes, edges)
    preds = get_predecessors(nodes, edges)

    for i in order
        best_r      = first(devices)
        best_finish = Inf
        best_start  = 0.0

        for r in devices
            t      = earliest_start(i, r, preds, assignment, start_times,
                                    runtime_matrix, device_free_at)
            finish = t + runtime_matrix[(i, r)]
            if finish < best_finish
                best_finish = finish
                best_r      = r
                best_start  = t
            end
        end

        assignment[i]          = best_r
        start_times[i]         = best_start
        device_free_at[best_r] = best_start + runtime_matrix[(i, best_r)]
    end

    makespan = maximum(start_times[i] + runtime_matrix[(i, assignment[i])]
                       for i in nodes)
    return assignment, start_times, makespan
end


# ============================================================
# 3. MIN-MIN SCHEDULER
# ============================================================


function minmin_scheduler(nodes, edges, runtime_matrix, devices)
    unscheduled    = Set(nodes)
    assignment     = Dict{Int, Symbol}()
    start_times    = Dict{Int, Float64}()
    device_free_at = Dict(d => 0.0 for d in devices)
    preds          = get_predecessors(nodes, edges)

    while !isempty(unscheduled)
        best_task        = nothing
        best_device      = nothing
        best_finish_time = Inf
        best_start_time  = 0.0

        for i in unscheduled
            any(!haskey(assignment, p) for p in preds[i]) && continue

            for r in devices
                if !haskey(runtime_matrix, (i, r))
                    error("Missing runtime entry for task $i on device $r")
                end
                t      = earliest_start(i, r, preds, assignment, start_times,
                                        runtime_matrix, device_free_at)
                finish = t + runtime_matrix[(i, r)]
                if finish < best_finish_time
                    best_finish_time = finish
                    best_task        = i
                    best_device      = r
                    best_start_time  = t
                end
            end
        end

        if best_task === nothing
            error("No schedulable task found — possible cycle or disconnected graph.")
        end

        assignment[best_task]       = best_device
        start_times[best_task]      = best_start_time
        device_free_at[best_device] = best_start_time + runtime_matrix[(best_task, best_device)]
        delete!(unscheduled, best_task)
    end

    makespan = maximum(start_times[i] + runtime_matrix[(i, assignment[i])]
                       for i in nodes)
    return assignment, start_times, makespan
end


# ============================================================
# 4. MAX-MIN SCHEDULER
# ============================================================


function maxmin_scheduler(nodes, edges, runtime_matrix, devices)
    unscheduled    = Set(nodes)
    assignment     = Dict{Int, Symbol}()
    start_times    = Dict{Int, Float64}()
    device_free_at = Dict(d => 0.0 for d in devices)
    preds          = get_predecessors(nodes, edges)

    while !isempty(unscheduled)
        best_task   = nothing
        best_device = nothing
        best_time   = -Inf
        best_start  = 0.0

        for i in unscheduled
            any(!haskey(assignment, p) for p in preds[i]) && continue

            min_finish = Inf
            min_device = nothing
            min_start  = 0.0

            for r in devices
                t      = earliest_start(i, r, preds, assignment, start_times,
                                        runtime_matrix, device_free_at)
                finish = t + runtime_matrix[(i, r)]
                if finish < min_finish
                    min_finish = finish
                    min_device = r
                    min_start  = t
                end
            end

            if min_finish > best_time
                best_time   = min_finish
                best_task   = i
                best_device = min_device
                best_start  = min_start
            end
        end

        if best_task === nothing
            error("No schedulable task found — possible cycle or disconnected graph.")
        end

        assignment[best_task]       = best_device
        start_times[best_task]      = best_start
        device_free_at[best_device] = best_start + runtime_matrix[(best_task, best_device)]
        delete!(unscheduled, best_task)
    end

    makespan = maximum(start_times[i] + runtime_matrix[(i, assignment[i])]
                       for i in nodes)
    return assignment, start_times, makespan
end


# ============================================================
# 5. ROUND-ROBIN SCHEDULER
# ============================================================


function round_robin_scheduler(nodes, edges, runtime_matrix, devices)
    assignment     = Dict{Int, Symbol}()
    start_times    = Dict{Int, Float64}()
    device_free_at = Dict(d => 0.0 for d in devices)

    order = topo_sort(nodes, edges)
    preds = get_predecessors(nodes, edges)

    d_idx = 1
    for i in order
        r     = devices[d_idx]
        d_idx = d_idx % length(devices) + 1
        t     = earliest_start(i, r, preds, assignment, start_times,
                                runtime_matrix, device_free_at)
        assignment[i]     = r
        start_times[i]    = t
        device_free_at[r] = t + runtime_matrix[(i, r)]
    end

    makespan = maximum(start_times[i] + runtime_matrix[(i, assignment[i])]
                       for i in nodes)
    return assignment, start_times, makespan
end


# ============================================================
# 6. HEFT SCHEDULER
# ============================================================


function heft_scheduler(nodes, edges, runtime_matrix, devices)
    preds = get_predecessors(nodes, edges)

    rank = Dict(
        i => mean(runtime_matrix[(i, r)] for r in devices)
        for i in nodes
    )

    priority_order = sort(nodes, by = i -> -rank[i])

    assignment     = Dict{Int, Symbol}()
    start_times    = Dict{Int, Float64}()
    device_free_at = Dict(d => 0.0 for d in devices)
    unscheduled    = Set(priority_order)

    while !isempty(unscheduled)
        scheduled_any = false

        for i in priority_order
            i ∉ unscheduled && continue
            any(!haskey(assignment, p) for p in preds[i]) && continue

            best_r      = first(devices)
            best_finish = Inf
            best_start  = 0.0

            for r in devices
                t      = earliest_start(i, r, preds, assignment, start_times,
                                        runtime_matrix, device_free_at)
                finish = t + runtime_matrix[(i, r)]
                if finish < best_finish
                    best_finish = finish
                    best_r      = r
                    best_start  = t
                end
            end

            assignment[i]          = best_r
            start_times[i]         = best_start
            device_free_at[best_r] = best_start + runtime_matrix[(i, best_r)]
            delete!(unscheduled, i)
            scheduled_any = true
        end

        if !scheduled_any
            error("HEFT: no progress made — possible cycle in the DAG.")
        end
    end

    makespan = maximum(start_times[i] + runtime_matrix[(i, assignment[i])]
                       for i in nodes)
    return assignment, start_times, makespan
end

end 
