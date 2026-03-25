

using Random
using Printf
using Statistics
using Dates

include(joinpath(@__DIR__, "..", "src", "dag_generator.jl"))
include(joinpath(@__DIR__, "..", "src", "runtime_prediction.jl"))
include(joinpath(@__DIR__, "..", "src", "milp_scheduler.jl"))
include(joinpath(@__DIR__, "..", "src", "heuristic_scheduler.jl"))
include(joinpath(@__DIR__, "..", "src", "evaluation.jl"))

using .DAGGenerator
using .RuntimePrediction
using .MILPScheduler
using .HeuristicSchedulers
using .Evaluation


Random.seed!(42)

const TASK_SIZES = [20, 40, 60, 80, 100]
const N_LAYERS   = 4
const EDGE_PROB  = 0.4
const N_RUNS     = 3        



all_results = Dict{Int, Vector{Dict}}()



println("\n" * "="^50)
println("DAGGER.JL TASK SCHEDULER BENCHMARK SUITE")
println("Started: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("="^50)

for n in TASK_SIZES
    println("\n" * "-"^40)
    println("Task size: $n")
    println("-"^40)

    size_results = Dict[]

    for run_id in 1:N_RUNS
        println("  Run $run_id / $N_RUNS ...")
        push!(size_results, Evaluation.run_single_experiment(n, N_LAYERS, EDGE_PROB))
    end

    all_results[n] = size_results

    # --------------------------------------------------------
    # Print averaged summary
    # --------------------------------------------------------
    methods = sort(collect(keys(size_results[1])), by=string)

    println("\n  Average Results (n = $n, $N_RUNS runs):")
    println("  " * "-"^60)
    @printf("  %-12s | %10s | %10s | %10s\n",
            "Scheduler", "Makespan", "Time (s)", "Speedup")
    println("  " * "-"^60)

    for method in methods
        makespans = [r[method][:makespan] for r in size_results]
        times     = [r[method][:time]     for r in size_results]
        speedups  = [r[method][:speedup_vs_HEFT] for r in size_results]

        @printf("  %-12s | %10.2f | %10.4f | %10.3f\n",
                string(method),
                mean(makespans),
                mean(times),
                mean(speedups))
    end
end

println("\n" * "="^50)
println("Benchmark complete.")
println("Finished: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("="^50)



function save_results(results::Dict, filename::String)
    mkpath(dirname(filename))
    open(filename, "w") do io
        println(io, "DAGGER.JL SCHEDULER BENCHMARK RESULTS")
        println(io, "Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)

        for n in sort(collect(keys(results)))
            println(io, "Task Size: $n")
            for (i, run) in enumerate(results[n])
                println(io, "  Run $i:")
                for (method, metrics) in run
                    @printf(io, "    %-12s  makespan=%8.2f  time=%8.4f s  speedup=%6.3f\n",
                            string(method),
                            metrics[:makespan],
                            metrics[:time],
                            metrics[:speedup_vs_HEFT])
                end
            end
            println(io)
        end
    end
    println("Results saved to: $filename")
end

save_path = joinpath(@__DIR__, "..", "results", "benchmark_results.txt")
save_results(all_results, save_path)
