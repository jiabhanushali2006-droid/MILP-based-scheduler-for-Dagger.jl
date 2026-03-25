module DAGGenerator

using Random

export generate_layered_dag,
       generate_pipeline_dag,
       generate_fork_join_dag,
       generate_tree_dag,
       generate_random_dag,
       generate_tgff_dag


# ============================================================
# Layered DAG Generator
# ============================================================

function generate_layered_dag(n_tasks::Int, n_layers::Int, edge_probability::Float64; seed=nothing)
    rng   = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    nodes = collect(1:n_tasks)
    edges = Tuple{Int,Int}[]

    # distribute tasks across layers
    layer_sizes = fill(div(n_tasks, n_layers), n_layers)
    layer_sizes[end] += n_tasks - sum(layer_sizes)

    layers = Vector{Vector{Int}}()

    index = 1
    for sz in layer_sizes
        push!(layers, nodes[index:(index + sz - 1)])
        index += sz
    end

    # create edges between consecutive layers
    for l in 1:(n_layers - 1)
        for i in layers[l]
            for j in layers[l + 1]
                if rand(rng) < edge_probability
                    push!(edges, (i, j))
                end
            end
        end
    end

    return nodes, edges, layers
end


# ============================================================
# Pipeline DAG
# ============================================================

function generate_pipeline_dag(n_tasks::Int)

    nodes = collect(1:n_tasks)
    edges = Tuple{Int,Int}[]

    for i in 1:(n_tasks - 1)
        push!(edges, (i, i + 1))
    end

    return nodes, edges
end


# ============================================================
# Fork-Join DAG
# ============================================================


function generate_fork_join_dag(n_branches::Int)

    nodes = collect(1:(n_branches + 2))
    edges = Tuple{Int,Int}[]

    start_node = 1
    end_node   = n_branches + 2

    for i in 2:(n_branches + 1)
        push!(edges, (start_node, i))
        push!(edges, (i, end_node))
    end

    return nodes, edges
end


# ============================================================
# Binary Tree DAG
# ============================================================


function generate_tree_dag(depth::Int)

    total_nodes = 2^depth - 1
    nodes = collect(1:total_nodes)
    edges = Tuple{Int,Int}[]

    for i in 1:div(total_nodes, 2)
        left  = 2 * i
        right = 2 * i + 1

        if left <= total_nodes
            push!(edges, (i, left))
        end
        if right <= total_nodes
            push!(edges, (i, right))
        end
    end

    return nodes, edges
end


# ============================================================
# Random DAG
# ============================================================


function generate_random_dag(n_tasks::Int, edge_probability::Float64)

    nodes = collect(1:n_tasks)
    edges = Tuple{Int,Int}[]

    for i in 1:n_tasks
        for j in (i + 1):n_tasks
            if rand() < edge_probability
                push!(edges, (i, j))
            end
        end
    end

    return nodes, edges
end


# ============================================================
# TGFF-style DAG Generator
# ============================================================

function generate_tgff_dag(n_tasks::Int, max_parallelism::Int, edge_density::Float64)

    nodes  = collect(1:n_tasks)
    edges  = Tuple{Int,Int}[]
    layers = Vector{Vector{Int}}()

    remaining    = n_tasks
    current_task = 1

    # build layers of random width
    while remaining > 0
        layer_size  = rand(1:min(max_parallelism, remaining))
        layer_nodes = nodes[current_task:(current_task + layer_size - 1)]
        push!(layers, layer_nodes)
        current_task += layer_size
        remaining    -= layer_size
    end

    # connect consecutive layers
    for l in 1:(length(layers) - 1)
        for u in layers[l]
            for v in layers[l + 1]
                if rand() < edge_density
                    push!(edges, (u, v))
                end
            end
        end
    end

    return nodes, edges, layers
end


# ============================================================
# Self-test / demo
# ============================================================


function example()
    println("=== Layered DAG ===")
    nodes, edges, layers = generate_layered_dag(50, 5, 0.3)
    println("Tasks: ", length(nodes), "  Edges: ", length(edges))

    println("\n=== Pipeline DAG ===")
    nodes, edges = generate_pipeline_dag(10)
    println("Edges: ", edges)

    println("\n=== Fork-Join DAG ===")
    nodes, edges = generate_fork_join_dag(4)
    println("Edges: ", edges)

    println("\n=== Tree DAG (depth 3) ===")
    nodes, edges = generate_tree_dag(3)
    println("Edges: ", edges)

    println("\n=== Random DAG ===")
    nodes, edges = generate_random_dag(20, 0.2)
    println("Edges: ", length(edges))

    println("\n=== TGFF DAG ===")
    nodes, edges, layers = generate_tgff_dag(30, 5, 0.4)
    println("Tasks: ", length(nodes), "  Edges: ", length(edges), "  Layers: ", length(layers))
end

end # module DAGGenerator
