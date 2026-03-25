module Evaluation

using Random
using Dates
using Statistics


using ..DAGGenerator
using ..RuntimePrediction
using ..MILPScheduler
using ..HeuristicSchedulers

export run_single_experiment,
       run_benchmark_suite


# ============================================================
# UTILITY: wall-clock timing wrapper
# ============================================================


function time_function(f)
    t0     = now()
    result = f()
    t1     = now()
    elapsed = (t1 - t0).value / 1e3   # Dates.value returns milliseconds
    return result, elapsed
end


# ============================================================
# METRIC 1: Makespan
# ============================================================


function compute_makespan(start_times, assignment, runtime_matrix, nodes)
    max_finish = 0.0
    for i in nodes
        haskey(start_times, i) || continue
        haskey(assignment,  i) || continue
        device = assignment[i]
        haskey(runtime_matrix, (i, device)) || continue
        max_finish = max(max_finish, start_times[i] + runtime_matrix[(i, device)])
    end
    return max_finish
end


# ============================================================
# METRIC 2: Resource Utilisation
# ============================================================


function compute_utilization(start_times, assignment, runtime_matrix, nodes, devices)
    device_busy = Dict(d => 0.0 for d in devices)

    for i in nodes
        haskey(assignment, i)  || continue
        r = assignment[i]
        haskey(runtime_matrix, (i, r)) || continue
        device_busy[r] += runtime_matrix[(i, r)]
    end

    makespan = compute_makespan(start_times, assignment, runtime_matrix, nodes)

    return Dict(
        d => (makespan == 0.0 ? 0.0 : device_busy[d] / makespan)
        for d in devices
    )
end


# ============================================================
# METRIC 3: Speedup
# ============================================================


function compute_speedup(baseline::Float64, metrics::Dict)
    ms = get(metrics, :makespan, Inf)
    return (ms == 0.0 || isinf(ms)) ? 0.0 : baseline / ms
end


# ============================================================
# SINGLE EXPERIMENT
# ============================================================


function run_single_experiment(n_tasks::Int, n_layers::Int, edge_prob::Float64)

    println("\n==============================")
    println("Experiment: $n_tasks tasks, $n_layers layers, p=$edge_prob")
    println("==============================")

    # --------------------------------------------------------
    # Step 1: DAG generation
    # --------------------------------------------------------
    nodes, edges, _ = DAGGenerator.generate_layered_dag(n_tasks, n_layers, edge_prob)

    # --------------------------------------------------------
    # Step 2: ML pipeline
    # --------------------------------------------------------
    features       = RuntimePrediction.generate_task_features(nodes)
    best_model     = RuntimePrediction.train_and_select_model()
    runtime_matrix = RuntimePrediction.build_runtime_matrix(nodes, features, best_model)

    devices = [:CPU, :GPU]
    results = Dict{Symbol, Dict}()

    # --------------------------------------------------------
    # MILP scheduler
    # --------------------------------------------------------
    try
        (milp_res, milp_time) = time_function(() ->
            MILPScheduler.solve_milp_scheduler(
                nodes, edges, features, runtime_matrix, devices
            )
        )

        assignment, start_times, makespan = milp_res

        if isempty(assignment)
            results[:MILP] = Dict(
                :makespan    => Inf,
                :time        => milp_time,
                :utilization => Dict(d => 0.0 for d in devices)
            )
        else
            utilization = compute_utilization(
                start_times, assignment, runtime_matrix, nodes, devices
            )
            results[:MILP] = Dict(
                :makespan    => makespan,
                :time        => milp_time,
                :utilization => utilization
            )
        end

    catch e
        @warn "MILP failed" exception=e
        results[:MILP] = Dict(
            :makespan    => Inf,
            :time        => Inf,
            :utilization => Dict(d => 0.0 for d in devices)
        )
    end

    # --------------------------------------------------------
    # Heuristic schedulers
    # --------------------------------------------------------
    schedulers = Dict(
        :HEFT       => HeuristicSchedulers.heft_scheduler,
        :MinMin     => HeuristicSchedulers.minmin_scheduler,
        :MaxMin     => HeuristicSchedulers.maxmin_scheduler,
        :Greedy     => HeuristicSchedulers.greedy_scheduler,
        :Random     => HeuristicSchedulers.random_scheduler,
        :RoundRobin => HeuristicSchedulers.round_robin_scheduler
    )

    for (name, scheduler) in schedulers
        try
            (res, t) = time_function(() ->
                scheduler(nodes, edges, runtime_matrix, devices)
            )

            assignment, start_times, makespan = res
            utilization = compute_utilization(
                start_times, assignment, runtime_matrix, nodes, devices
            )

            results[name] = Dict(
                :makespan    => makespan,
                :time        => t,
                :utilization => utilization
            )

        catch e
            @warn "Scheduler $name failed" exception=e
            results[name] = Dict(
                :makespan    => Inf,
                :time        => Inf,
                :utilization => Dict(d => 0.0 for d in devices)
            )
        end
    end

    # --------------------------------------------------------
    # Speedup relative to HEFT
    # --------------------------------------------------------
    baseline = get(get(results, :HEFT, Dict()), :makespan, Inf)

    for (method, metrics) in results
        metrics[:speedup_vs_HEFT] = compute_speedup(baseline, metrics)
    end

    return results
end


# ============================================================
# BENCHMARK SUITE
# ============================================================


function run_benchmark_suite()
    Random.seed!(42)

    task_sizes = [20, 50, 80, 100]
    all_results = Dict{Int, Dict}()

    for n in task_sizes
        results = run_single_experiment(n, 4, 0.4)
        all_results[n] = results

        println("\n----- Results for $n tasks -----")
        for (method, metrics) in results
            ms  = round(metrics[:makespan],          digits=2)
            t   = round(metrics[:time],              digits=4)
            spd = round(metrics[:speedup_vs_HEFT],   digits=3)
            println("$method | makespan: $ms | time: $t s | speedup vs HEFT: $spd")
        end
    end

    return all_results
end

end 
