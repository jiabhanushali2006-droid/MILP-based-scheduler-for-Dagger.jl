module MILPScheduler

using JuMP
using HiGHS

export solve_milp_scheduler


const MILP_TIME_LIMIT_SEC = 20.0

function solve_milp_scheduler(nodes, edges, features, runtime_matrix, devices;
                               time_limit::Float64 = MILP_TIME_LIMIT_SEC)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Hard time budget — returns best incumbent when limit is hit.
    # Without this, n≥15 instances can run for minutes or never finish.
    set_attribute(model, "time_limit", time_limit)

    T = nodes
    K = devices
    E = edges

    # -------------------------------------------------------
    # Big-M: upper bound on any start / completion time
    # -------------------------------------------------------
    BIG_M = sum(
        maximum(runtime_matrix[(i, k)] for k in K)
        for i in T
    )

    # -------------------------------------------------------
    # Decision variables
    # -------------------------------------------------------
    @variable(model, x[T, K], Bin)
    @variable(model, S[T]    >= 0)
    @variable(model, C[T]    >= 0)
    @variable(model, o[T, T, K], Bin)
    @variable(model, Cmax    >= 0)

    # -------------------------------------------------------
    # Objective
    # -------------------------------------------------------
    @objective(model, Min, Cmax)

    # -------------------------------------------------------
    # (1) Assignment: each task on exactly one device
    # -------------------------------------------------------
    @constraint(model, assign[i in T],
        sum(x[i, k] for k in K) == 1
    )

    # -------------------------------------------------------
    # (2) Completion time definition
    # -------------------------------------------------------
    @constraint(model, completion[i in T],
        C[i] == S[i] + sum(runtime_matrix[(i, k)] * x[i, k] for k in K)
    )

    # -------------------------------------------------------
    # (3) Precedence constraints
    # -------------------------------------------------------
    for (i, j) in E
        @constraint(model, S[j] >= C[i])
    end

    # -------------------------------------------------------
    # (4) Machine non-overlap (disjunctive / Big-M)
    # -------------------------------------------------------
    for i in T, j in T, k in K
        if i < j   
            @constraint(model, o[i, j, k] <= x[i, k])
            @constraint(model, o[i, j, k] <= x[j, k])
            @constraint(model, o[j, i, k] <= x[i, k])
            @constraint(model, o[j, i, k] <= x[j, k])

            # exactly one ordering when both on same device
            @constraint(model,
                o[i, j, k] + o[j, i, k] >= x[i, k] + x[j, k] - 1
            )

            # enforce ordering: i before j
            @constraint(model,
                S[j] >= S[i] + runtime_matrix[(i, k)] - BIG_M * (1 - o[i, j, k])
            )

            # enforce ordering: j before i
            @constraint(model,
                S[i] >= S[j] + runtime_matrix[(j, k)] - BIG_M * (1 - o[j, i, k])
            )
        end
    end

    # -------------------------------------------------------
    # (5) Makespan lower bound
    # -------------------------------------------------------
    @constraint(model, cmax_lb[i in T], Cmax >= C[i])

    # -------------------------------------------------------
    # Solve
    # -------------------------------------------------------
    optimize!(model)

    status = termination_status(model)

    
    if !has_values(model)
        error("MILP returned $status with no feasible solution. " *
              "Try increasing time_limit or reducing n.")
    end

    if status == MOI.OPTIMAL
        # nothing extra to report
    elseif status == MOI.TIME_LIMIT
        obj_bound = objective_bound(model)
        obj_value = objective_value(model)
        gap_pct   = abs(obj_value - obj_bound) / max(abs(obj_value), 1e-8) * 100
        @info "MILP hit time limit ($(time_limit)s). " *
              "Returning best incumbent (gap ≈ $(round(gap_pct, digits=1))%). " *
              "Solution is suboptimal but valid."
    else
        @warn "MILP solver returned status: $status — solution may be unreliable."
    end

    # -------------------------------------------------------
    # Extract solution (works for both OPTIMAL and TIME_LIMIT incumbents)
    # -------------------------------------------------------
    assignment  = Dict{Int, Symbol}()
    start_times = Dict{Int, Float64}()

    for i in T
        start_times[i] = value(S[i])
        for k in K
            if value(x[i, k]) > 0.5
                assignment[i] = k
            end
        end
    end

    return assignment, start_times, value(Cmax)
end

end
