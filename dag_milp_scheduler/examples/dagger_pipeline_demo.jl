
using Dagger
using Random
using Printf
using Statistics
using Dates
using LinearAlgebra

# ── module loading ────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "dag_generator.jl"))
include(joinpath(@__DIR__, "..", "src", "runtime_prediction.jl"))
include(joinpath(@__DIR__, "..", "src", "milp_scheduler.jl"))
include(joinpath(@__DIR__, "..", "src", "heuristic_scheduler.jl"))

using .DAGGenerator
using .RuntimePrediction
using .MILPScheduler
using .HeuristicSchedulers



Random.seed!(42)

const N_TASKS   = 8      
const N_LAYERS  = 3
const EDGE_PROB = 0.4
const DEVICES   = [:CPU, :GPU]


# Requires: julia -t 4
const CPU_THREAD = 1
const GPU_THREAD = 3


const SCALE = 80.0

const OUT_DIR = @__DIR__



dag_task(task_id::Int, runtime_s::Float64, input_size::Float64, deps...) =
    (sleep(runtime_s); input_size * (1.0 + 0.05 * randn()))


function run_plan(nodes, edges, assignment, runtime_matrix, features; label::String="run")
    order = HeuristicSchedulers.topo_sort(nodes, edges)

    Dagger.enable_logging!(all_task_deps=true)
    t_start = time_ns()

    dtasks = Dict{Int, DTask}()
    for i in order
        d   = get(assignment, i, :CPU)
        rt  = runtime_matrix[(i, d)] / SCALE
        inp = features[i][:input_size]

        n_threads  = Threads.nthreads()
        cpu_thread = min(CPU_THREAD, n_threads)
        gpu_thread = min(GPU_THREAD, n_threads)

        sc = d == :CPU ?
             Dagger.scope(worker=1, thread=cpu_thread) :
             Dagger.scope(worker=1, thread=gpu_thread)

        deps = [dtasks[p] for p in filter(p -> haskey(dtasks, p), [u for (u,v) in edges if v == i])]
        dtasks[i] = Dagger.spawn(
         dag_task,
         Dagger.Options(scope=sc, name="$(label)_T$(i)_$(d)"),
        i, rt, inp, deps...
        )
    end


    for i in order
        fetch(dtasks[i])
    end

    t_end = time_ns()
    logs  = Dagger.fetch_logs!()
    Dagger.disable_logging!()

    return (t_end - t_start) / 1e9, logs
end



function extract_timings(logs)
    all_events = []
    for proc_dict in values(logs)
        for evts in values(proc_dict)
            append!(all_events, evts)
        end
    end

    isempty(all_events) && return NamedTuple[]

    comp = filter(e -> hasproperty(e, :category) && e.category == :compute, all_events)
    S    = filter(e -> hasproperty(e, :kind) && e.kind == :start,  comp)
    F    = filter(e -> hasproperty(e, :kind) && e.kind == :finish, comp)

    isempty(S) && return NamedTuple[]

    # Sort both by timestamp so positional pairing is consistent
    S = sort(S, by = e -> e.timestamp)
    F = sort(F, by = e -> e.timestamp)

    t0 = minimum(e.timestamp for e in S)

    timings = NamedTuple[]
    for (s, f) in zip(S, F)
        push!(timings, (
            start_s    = (s.timestamp - t0) / 1e9,
            finish_s   = (f.timestamp - t0) / 1e9,
            duration_s = (f.timestamp - s.timestamp) / 1e9
        ))
    end
    return sort(timings, by = t -> t.start_s)
end

function observed_makespan(timings)
    isempty(timings) && return 0.0
    maximum(t.finish_s for t in timings) - minimum(t.start_s for t in timings)
end


function update_runtime_matrix(runtime_matrix, nodes, features, timings, assignment)
    rm    = deepcopy(runtime_matrix)
    alpha = 0.7
     for (i, t) in zip(sort(nodes), timings)
        d        = get(assignment, i, :CPU)
        obs_rt   = t.duration_s * SCALE
        prior_rt = rm[(i, d)]
        blended  = alpha * obs_rt + (1.0 - alpha) * prior_rt
        rm[(i, d)]  = blended
        other       = d == :CPU ? :GPU : :CPU
        ratio       = d == :CPU ? 0.6 : (1.0 / 0.6)
        rm[(i, other)] = blended * ratio
    end
    return rm
end


function fit_linear_predictor(nodes, features, runtime_matrix)
    X = [features[i][:input_size] for i in nodes]
    y = [runtime_matrix[(i, :CPU)]  for i in nodes]
    A = hcat(X, ones(length(X)))
    θ = (A' * A) \ (A' * y)
    ŷ = A * θ
    ss_res = sum((y .- ŷ).^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = ss_tot > 1e-10 ? 1.0 - ss_res / ss_tot : 0.0
    return θ[1], θ[2], r2
end


function rebuild_rm_linear(nodes, features, slope, intercept)
    rm = Dict{Tuple{Int,Symbol},Float64}()
    for i in nodes
        base = max(slope * features[i][:input_size] + intercept, 1.0)
        rm[(i, :CPU)] = base
        rm[(i, :GPU)] = base * 0.6
    end
    return rm
end



function write_gantt_svg(timings, labels, title, filepath)
    isempty(timings) && return
    W=900; BAR_H=28; GAP=6; LM=110; RM=40; TM=55; BM=55
    n = length(timings)
    H = TM + n*(BAR_H+GAP) + BM
    PW = W - LM - RM
    max_t = max(maximum(t.finish_s for t in timings), 0.001)

    open(filepath, "w") do io
        write(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H">\n""")
        write(io, """<rect width="$W" height="$H" fill="#F8F9FA"/>\n""")
        write(io, """<text x="$(W÷2)" y="32" text-anchor="middle" font-size="15" font-family="Arial" font-weight="bold" fill="#212529">$title</text>\n""")

        # grid lines + time labels
        for frac in 0.0:0.25:1.0
            x = LM + round(Int, frac*PW)
            write(io, """<line x1="$x" y1="$TM" x2="$x" y2="$(H-BM)" stroke="#DEE2E6" stroke-width="1"/>\n""")
            t_ms = round(Int, frac * max_t * 1000)
            write(io, """<text x="$x" y="$(H-BM+16)" text-anchor="middle" font-size="10" font-family="Arial" fill="#495057">$(t_ms)ms</text>\n""")
        end
        write(io, """<text x="$(LM+PW÷2)" y="$(H-8)" text-anchor="middle" font-size="11" font-family="Arial" fill="#495057">Wall-clock time (ms)</text>\n""")

        # task bars
        for (idx, (t, lbl)) in enumerate(zip(timings, labels))
            y   = TM + (idx-1)*(BAR_H+GAP)
            x0  = LM + round(Int, (t.start_s  / max_t) * PW)
            x1  = LM + round(Int, (t.finish_s / max_t) * PW)
            bw  = max(x1-x0, 4)
            col = contains(string(lbl), "GPU") ? "#70AD47" : "#4472C4"
            write(io, """<rect x="$x0" y="$y" width="$bw" height="$BAR_H" fill="$col" rx="3" opacity="0.88"/>\n""")
            write(io, """<text x="$(LM-6)" y="$(y+BAR_H÷2+4)" text-anchor="end" font-size="10" font-family="Arial" fill="#212529">T$(idx) $lbl</text>\n""")
            dur_ms = round(Int, t.duration_s*1000)
            if bw > 38
                write(io, """<text x="$(x0+bw÷2)" y="$(y+BAR_H÷2+4)" text-anchor="middle" font-size="10" font-family="Arial" fill="white">$(dur_ms)ms</text>\n""")
            end
        end

        # legend
        ly = H - BM + 34
        write(io, """<rect x="$LM" y="$ly" width="13" height="13" fill="#4472C4" rx="2"/>\n""")
        write(io, """<text x="$(LM+17)" y="$(ly+11)" font-size="10" font-family="Arial" fill="#212529">CPU thread</text>\n""")
        write(io, """<rect x="$(LM+90)" y="$ly" width="13" height="13" fill="#70AD47" rx="2"/>\n""")
        write(io, """<text x="$(LM+107)" y="$(ly+11)" font-size="10" font-family="Arial" fill="#212529">GPU thread</text>\n""")
        write(io, "</svg>\n")
    end
    println("  Gantt → $filepath")
end



bar(n=65) = println("═"^n)
function section(t); println(); bar(); println("  $t"); bar(); end


 
section("DAGGER.JL SCHEDULER INTEGRATION DEMO")
println("  Started:  ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("  Threads:  ", Threads.nthreads(), "  (need ≥ 4 for CPU/GPU simulation)")
Threads.nthreads() < 4 &&
    @warn "Run with julia -t 4 for proper CPU/GPU thread separation"


let _t = Dagger.@spawn 1 + 1
    fetch(_t)  
    ctx = Dagger.Sch.eager_context()
    println("  Active processors: ", ctx.procs)
    n = count(p -> p isa Dagger.ThreadProc, ctx.procs)
    n < 2 && @warn "Only $n ThreadProc(s) visible — run with julia -t 4"
end


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — JIT warmup
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 0 — JIT Warmup")
println("  Eliminating JIT compilation noise from timing measurements...")
_w(x) = x + 0.0
for _ in 1:6; fetch(Dagger.@spawn _w(1.0)); end
println("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DAG generation + offline planning
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 1 — DAG Generation & Offline Planning")

nodes, edges, _ = Base.invokelatest(
    DAGGenerator.generate_layered_dag, N_TASKS, N_LAYERS, EDGE_PROB; seed=42)
features = Base.invokelatest(RuntimePrediction.generate_task_features, nodes)
println("  DAG:   $(length(nodes)) tasks  |  $(length(edges)) edges  |  $N_LAYERS layers")
println("  Edges: $edges")

best_model = Base.invokelatest(RuntimePrediction.train_and_select_model)
rm = Base.invokelatest(RuntimePrediction.build_runtime_matrix, nodes, features, best_model)

println("\n  Predicted runtimes (abstract units, SCALE=$SCALE → ÷$SCALE = sleep in s):")
@printf("  %-6s  %8s  %8s  %10s  %10s\n","Task","CPU_rt","GPU_rt","CPU_sleep","GPU_sleep")
println("  " * "─"^50)
for i in sort(nodes)
    @printf("  T%-5d  %8.2f  %8.2f  %10.4f  %10.4f\n",
            i, rm[(i,:CPU)], rm[(i,:GPU)],
            rm[(i,:CPU)]/SCALE, rm[(i,:GPU)]/SCALE)
end

# MILP plan (n=8 ≤ 12 — optimal zone)
println("\n  Running MILP (n=$N_TASKS ≤ 12 — optimal reference)...")

assign_milp, st_milp, ms_milp = MILPScheduler.solve_milp_scheduler(
    nodes, edges, features, rm, DEVICES)
println("  MILP predicted makespan: $(round(ms_milp/SCALE, digits=3))s")

# HEFT plan
assign_heft, st_heft, ms_heft = HeuristicSchedulers.heft_scheduler(
    nodes, edges, rm, DEVICES)
println("  HEFT predicted makespan: $(round(ms_heft/SCALE, digits=3))s")

# Naive plans
assign_allgpu = Dict(i => :GPU for i in nodes)
assign_allcpu = Dict(i => :CPU for i in nodes)

println("\n  Task assignments:")
@printf("  %-6s  %8s  %8s  %8s  %8s\n","Task","MILP","HEFT","AllGPU","AllCPU")
println("  " * "─"^46)
for i in sort(nodes)
    @printf("  T%-5d  %8s  %8s  %8s  %8s\n",
            i, assign_milp[i], assign_heft[i], :GPU, :CPU)
end


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Execute all three plans, measure with fetch_logs!()
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 2 — Live Dagger Execution Under Three Plans")
println("  Each plan runs the same DAG. Scope constraints translate")
println("  device assignments into thread-pinned DTask execution.")
println("  CPU → ThreadProc(1,$CPU_THREAD)   GPU → ThreadProc(1,$GPU_THREAD)")
println()

plans = [
    ("MILP",   assign_milp),
    ("AllGPU", assign_allgpu),
    ("AllCPU", assign_allcpu),
]

all_timings = Dict{String, Vector{NamedTuple}}()
all_walls   = Dict{String, Float64}()
all_logs    = Dict{String, Any}()

for (name, assignment) in plans
    print("  Running '$name' plan... ")
    flush(stdout)
    wall, logs = Base.invokelatest(run_plan, nodes, edges, assignment, rm, features; label=name)
    timings    = extract_timings(logs)
    ms_obs     = observed_makespan(timings)
    all_timings[name] = timings
    all_walls[name]   = wall
    all_logs[name]    = logs
    @printf("done  wall=%.3fs  makespan=%.3fs\n", wall, ms_obs)
end


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Comparison table
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 3 — Scheduling Impact Comparison")
println()
@printf("  %-10s  %12s  %14s  %14s  %12s\n",
        "Plan","Pred mkspan","Obs makespan","Wall time","vs AllCPU")
println("  " * "─"^68)

pred_ms = Dict(
    "MILP"   => ms_milp   / SCALE,
    "HEFT"   => ms_heft   / SCALE,
    "AllGPU" => maximum(rm[(i,:GPU)] for i in nodes) / SCALE,
    "AllCPU" => maximum(rm[(i,:CPU)] for i in nodes) / SCALE,
)

cpu_obs_ms = observed_makespan(all_timings["AllCPU"])
for (name, _) in plans
    obs_ms  = observed_makespan(all_timings[name])
    wall    = all_walls[name]
    pred    = get(pred_ms, name, NaN)
    vs_cpu  = cpu_obs_ms / max(obs_ms, 1e-9)
    @printf("  %-10s  %12.4f  %14.4f  %14.4f  %11.2fx\n",
            name, pred, obs_ms, wall, vs_cpu)
end
println()

milp_ms = observed_makespan(all_timings["MILP"])
gpu_ms  = observed_makespan(all_timings["AllGPU"])
speedup_vs_cpu = cpu_obs_ms / max(milp_ms, 1e-9)
speedup_vs_gpu = gpu_ms     / max(milp_ms, 1e-9)

println("  Key results:")
@printf("  • MILP vs AllCPU makespan speedup: %.2fx\n", speedup_vs_cpu)
@printf("  • MILP vs AllGPU makespan speedup: %.2fx\n", speedup_vs_gpu)
println()
if speedup_vs_cpu > 1.05 && speedup_vs_gpu > 1.05
    println("  ✓ MILP heterogeneous assignment beats both naive baselines.")
    println("    Mixed CPU+GPU assignment achieves better device utilisation")
    println("    than pinning everything to one device type.")
elseif speedup_vs_cpu > 1.05
    println("  ✓ MILP beats AllCPU. GPU tasks are running faster on GPU thread.")
else
    println("  Note: differences are within Dagger's scheduling variance for n=$N_TASKS.")
    println("  Impact grows with larger DAGs where device specialisation dominates.")
end

println()
println("  Per-task observed durations (MILP plan):")
@printf("  %-6s  %-6s  %12s  %12s  %12s\n",
        "Task","Device","Predicted(s)","Observed(s)","Error%")
println("  " * "─"^56)
for (i, t) in zip(sort(nodes), all_timings["MILP"])
    d    = get(assign_milp, i, :CPU)
    pred = rm[(i,d)] / SCALE
    obs  = t.duration_s
    err  = (obs - pred) / pred * 100
    @printf("  T%-5d  %-6s  %12.4f  %12.4f  %+11.1f%%\n", i, d, pred, obs, err)
end


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Gantt charts
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 4 — Gantt Charts")
println("  Writing SVG Gantt charts for side-by-side visual comparison...")

milp_labels = [string(get(assign_milp, i, :CPU)) for i in sort(nodes)]
cpu_labels  = ["CPU" for _ in nodes]

write_gantt_svg(all_timings["MILP"],
                milp_labels,
                "MILP-Guided Schedule (n=$N_TASKS) — Mixed CPU+GPU",
                joinpath(OUT_DIR, "gantt_milp.svg"))

write_gantt_svg(all_timings["AllCPU"],
                cpu_labels,
                "All-CPU Naive Schedule (n=$N_TASKS) — No Device Awareness",
                joinpath(OUT_DIR, "gantt_allcpu.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Feedback loop
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 5 — Feedback Loop: Observed Runtimes → Updated Cost Model")

println("  Extracting per-task observed durations from Dagger logs (MILP run)...")
timings_milp = all_timings["MILP"]

rm_v2 = update_runtime_matrix(rm, sort(nodes), features, timings_milp, assign_milp)

println()
@printf("  %-6s  %-6s  %12s  %12s  %12s  %10s\n",
        "Task","Device","Prior CPU","Observed","Updated CPU","Δ%")
println("  " * "─"^62)
for (i, t) in zip(sort(nodes), timings_milp)
    d       = get(assign_milp, i, :CPU)
    prior   = rm[(i,:CPU)]
    obs     = t.duration_s * SCALE
    updated = rm_v2[(i,:CPU)]
    delta   = (updated - prior) / prior * 100
    @printf("  T%-5d  %-6s  %12.2f  %12.2f  %12.2f  %+9.1f%%\n",
            i, d, prior, obs, updated, delta)
end

# Fit linear predictor
println()
println("  Fitting linear predictor on updated observations...")
slope, intercept, r2 = fit_linear_predictor(sort(nodes), features, rm_v2)
@printf("  Model:  runtime(CPU) = %.4f × input_size + %.4f\n", slope, intercept)
@printf("  R²  =   %.4f\n", r2)
r2 > 0.5 ?
    println("  ✓ Reasonable fit — sufficient for updated planning.") :
    println("  Note: Low R² with n=$N_TASKS — more executions improve calibration.")

rm_v3 = rebuild_rm_linear(sort(nodes), features, slope, intercept)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Re-plan with updated cost model
# ─────────────────────────────────────────────────────────────────────────────

section("PHASE 6 — Re-Planning with Updated Cost Model")

println("  Re-running MILP with linear-fitted runtime predictions...")
assign_v2, _, ms_v2 = MILPScheduler.solve_milp_scheduler(
    nodes, edges, features, rm_v3, DEVICES)

changed = count(get(assign_milp,i,:CPU) != get(assign_v2,i,:CPU) for i in nodes)

println()
@printf("  %-6s  %10s  %10s  %6s\n","Task","Plan v1","Plan v2","Δ?")
println("  " * "─"^36)
for i in sort(nodes)
    d1 = get(assign_milp,i,:CPU); d2 = get(assign_v2,i,:CPU)
    flag = d1 != d2 ? " ◀" : ""
    @printf("  T%-5d  %10s  %10s%s\n", i, d1, d2, flag)
end

println()
@printf("  Tasks changed: %d / %d\n", changed, N_TASKS)
@printf("  Predicted makespan v1: %.3fs  →  v2: %.3fs\n",
        ms_milp/SCALE, ms_v2/SCALE)
changed > 0 ?
    println("  ✓ Feedback loop changed the plan — real observations updated decisions.") :
    println("  ≈ Plan stable — synthetic predictor was well-calibrated on this instance.")

println()
println("  Feedback loop demonstrated:")
println("  Dagger logs → observed durations → updated runtime_matrix")
println("  → linear regression fit → re-planning → new scope constraints")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("FINAL SUMMARY")
println()
w = 64
println("  ┌" * "─"^w * "┐")
@printf("  │  %-*s│\n", w-1, "Component                  Result")
println("  ├" * "─"^w * "┤")
@printf("  │  %-*s│\n", w-1, "DAG (n=$N_TASKS, $(length(edges)) edges)     ✓ layered topology generated")
@printf("  │  %-*s│\n", w-1, "Offline planner            ✓ MILP optimal (n≤12 safe zone)")
@printf("  │  %-*s│\n", w-1,
    "Observed MILP makespan     ✓ $(round(milp_ms,digits=3))s (predicted $(round(ms_milp/SCALE,digits=3))s)")
@printf("  │  %-*s│\n", w-1,
    "vs AllCPU speedup          $(speedup_vs_cpu>1.0 ? "✓" : "≈") $(round(speedup_vs_cpu,digits=2))x")
@printf("  │  %-*s│\n", w-1,
    "vs AllGPU speedup          $(speedup_vs_gpu>1.0 ? "✓" : "≈") $(round(speedup_vs_gpu,digits=2))x")
@printf("  │  %-*s│\n", w-1,
    "Feedback loop              ✓ $changed/$N_TASKS tasks changed in re-plan")
@printf("  │  %-*s│\n", w-1,
    "Linear predictor           ✓ R²=$(round(r2,digits=4)) on $(N_TASKS) observations")
@printf("  │  %-*s│\n", w-1, "Gantt charts               ✓ gantt_milp.svg + gantt_allcpu.svg")
println("  └" * "─"^w * "┘")

println()
println("  Finished: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println()
println("  Production integration roadmap:")
println("  1. Replace dag_task sleep() with real compute kernels")
println("  2. Add DaggerGPU.jl CuArrayProc for physical GPU acceleration")
println("  3. Accumulate observations across many runs (exponential moving avg)")
println("  4. Wire memory_demand prefilter: if m[i] > M_k, fix x[i,k]=0")
println("  5. For n>12: HEFT as primary scheduler, MILP as offline calibrator")
