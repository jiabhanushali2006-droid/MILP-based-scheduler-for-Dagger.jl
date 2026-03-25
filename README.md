# MILP-based-scheduler-for-Dagger.jl
# Hybrid DAG Scheduling using MILP and Heuristics

This repository implements a **Mixed Integer Linear Programming (MILP)-based scheduler** for DAG (Directed Acyclic Graph) workflows, along with a **hybrid scheduling strategy** that combines exact optimization with heuristic methods.

Developed as part of my **Google Summer of Code 2025 proposal** for Dagger.jl.

- MILP-based optimal scheduler for small DAGs
- Benchmarking against standard heuristics:
  - HEFT
  - MinMin / MaxMin
  - Greedy / Random
- Hybrid scheduler based on task-size thresholds
- Integration with Dagger.jl execution model
- Runtime prediction module for task execution

---

## Key Insight

- MILP achieves optimal schedules for small DAGs (n ≤ 12)
- Heuristics perform near-optimally for larger DAGs
- Hybrid approach balances **optimality and scalability**

---

##  Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
