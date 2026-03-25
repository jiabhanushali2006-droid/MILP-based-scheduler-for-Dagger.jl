module RuntimePrediction

using Random
using Statistics

Random.seed!(42)

export generate_task_features,
       generate_training_data,
       evaluate_models,
       select_best_model,
       train_and_select_model,
       build_runtime_matrix




function generate_task_features(nodes, rng=Random.default_rng())
    features = Dict{Int, Dict}()
    for i in nodes
        features[i] = Dict(
            :input_size    => rand(rng,1.0:1.0:100.0),
            :memory_demand => rand(rng,1.0:1.0:16.0)
        )
    end
    return features
end





function generate_training_data(n_samples::Int, rng=Random.default_rng())
    X = rand(rng,  10.0:1.0:100.0, n_samples)
    y = 2.0 .+ 0.5 .* X .+ 0.01 .* X .^ 2 .+ randn(n_samples)
    return X, y
end



linear_model(x)     = 2.0 + 0.5 * x
polynomial_model(x) = 2.0 + 0.5 * x + 0.01 * x^2

function tree_model(x)
    return x < 50 ? 20.0 + 0.3 * x : 10.0 + 0.8 * x
end




MAE(y_true, y_pred) = mean(abs.(y_true .- y_pred))

relative_error(y_true, y_pred) = mean(abs.(y_true .- y_pred) ./ max.(abs.(y_true), 1e-8))


function ranking_accuracy(y_true, y_pred)
    correct = 0
    total   = 0
    n = length(y_true)
    for i in 1:n
        for j in (i + 1):n
            if (y_true[i] < y_true[j]) == (y_pred[i] < y_pred[j])
                correct += 1
            end
            total += 1
        end
    end
    return total == 0 ? 0.0 : correct / total
end



function evaluate_models(X, y)
    models = Dict(
        :linear => x -> linear_model(x),
        :poly   => x -> polynomial_model(x),
        :tree   => x -> tree_model(x)
    )

    results = Dict()
    for (name, model) in models
        preds = [model(x) for x in X]
        results[name] = Dict(
            :MAE              => MAE(y, preds),
            :relative_error   => relative_error(y, preds),
            :ranking_accuracy => ranking_accuracy(y, preds)
        )
    end
    return results
end



function select_best_model(results)
    best_model = nothing
    best_score = -Inf
    for (name, metrics) in results
        score = metrics[:ranking_accuracy]
        if score > best_score
            best_score = score
            best_model = name
        end
    end
    return best_model
end


function train_and_select_model()
    X, y        = Base.invokelatest(generate_training_data,100)
    results     = Base.invokelatest(evaluate_models,X, y)
    best_model  = Base.invokelatest(select_best_model,results)
    println("Best model selected: ", best_model)
    return best_model
end



function build_runtime_matrix(nodes, features, best_model::Symbol)
    device_factors = Dict(:CPU => 1.0, :GPU => 0.6)
    devices        = collect(keys(device_factors))

    runtime_matrix = Dict{Tuple{Int, Symbol}, Float64}()

    for i in nodes
        input_size = features[i][:input_size]

        base = if best_model == :linear
            linear_model(input_size)
        elseif best_model == :poly
            polynomial_model(input_size)
        else
            tree_model(input_size)
        end

        for r in devices
            runtime_matrix[(i, r)] = device_factors[r] * base
        end
    end

    return runtime_matrix
end

end
