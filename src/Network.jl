module BioNetworks #
    using ..Bacterias
    using Catalyst
    using LinearAlgebra 
    using Plots

    export BioNetwork, add_bacterium!, build_edges!, plot_bionetwork, plot_bionetwork_3d, build_network_square!, build_network_cube!, assign_tetrahedral_roles

    mutable struct BioNetwork
        nodes::Dict{Int, Bacterium}
        edges::Vector{Tuple{Int, Int, Float64}} 
        d_max::Float64
        n_segments::Int 

        function BioNetwork(d_max, n_segments)
            new(Dict{Int, Bacterium}(), [], d_max, n_segments)
        end
    end

    function add_bacterium!(net::BioNetwork, b::Bacterium)
        net.nodes[b.id] = b
    end

    function build_edges!(net::BioNetwork)
        empty!(net.edges)
        ids = collect(keys(net.nodes))

        for i in 1:length(ids)
            for j in 1:length(ids)
                id1, id2 = ids[i], ids[j]
                
                if id1 == id2 
                    if !any(e -> e[1] == id1 && e[2] == id2, net.edges)  # seulement si le lien n'existe pas déjà
                        
            
                        push!(net.edges, (id1, id2, 0.0))  # Auto-connexion
                    end
                
                else
                    # Calcul de la distance entre les deux bactéries
                    dist = norm(net.nodes[id1].pos - net.nodes[id2].pos)

                    if dist <= net.d_max
                        push!(net.edges, (id1, id2, dist))
                    end
                end


  
            end
        end
        println("Graphe construit : $(length(net.edges)) liens créés.")
    end

    function build_network_cube!(net, n_bacteries, d_max, n_segments, circuit, params_dict, u0_dict, mode; sigma=0.5e-6)
        grid_size = ceil(Int, n_bacteries^(1/3))
        spacing = d_max / n_segments

        id = 1
        for i in 0:(grid_size-1)
            for j in 0:(grid_size-1)
                for k in 0:(grid_size-1)
                    if id > n_bacteries
                        break
                    end
                    pos = [i * spacing + randn() * sigma,
                           j * spacing + randn() * sigma,
                           k * spacing + randn() * sigma]
                    b = Bacterium(id, pos, circuit, params_dict, u0_dict; mode=mode)
                    add_bacterium!(net, b)
                    id += 1
                end
            end
        end

        build_edges!(net)
        return net
    end

    function build_network_square!(net,n_bacteries, d_max, n_segments,circuit, params_dict, u0_dict,mode;sigma=0.1)
        grid_size = ceil(Int, sqrt(n_bacteries))
        spacing = d_max / n_segments


        id = 1
        for i in 0:(grid_size-1)
            for j in 0:(grid_size-1)
                if id > n_bacteries
                    break
                end
                pos = [i * spacing + randn() * sigma, j * spacing + randn() * sigma]
                b = Bacterium(id, pos, circuit, params_dict, u0_dict; mode=mode)
                add_bacterium!(net, b)
                id += 1
            end
        end

        build_edges!(net)
        return net
    end

    function assign_tetrahedral_roles(net::BioNetwork)
        ids = collect(keys(net.nodes))

        # Bounding box
        min_x = minimum(net.nodes[id].pos[1] for id in ids)
        max_x = maximum(net.nodes[id].pos[1] for id in ids)
        min_y = minimum(net.nodes[id].pos[2] for id in ids)
        max_y = maximum(net.nodes[id].pos[2] for id in ids)
        min_z = minimum(net.nodes[id].pos[3] for id in ids)
        max_z = maximum(net.nodes[id].pos[3] for id in ids)

        # 4 coins tétraédriques fixes (inscrits dans le cube) — assignation fixe
        targets = [
            (:Input_A,  [min_x, min_y, min_z]),
            (:Input_B,  [max_x, max_y, min_z]),
            (:Output_0, [max_x, min_y, max_z]),
            (:Output_1, [min_x, max_y, max_z]),
        ]

        node_roles = Dict{Symbol, Int}()
        used_ids   = Set{Int}()

        for (role, target) in targets
            best_id   = -1
            best_dist = Inf
            for id in ids
                id in used_ids && continue
                d = norm(net.nodes[id].pos - target)
                if d < best_dist
                    best_dist = d
                    best_id   = id
                end
            end
            node_roles[role] = best_id
            push!(used_ids, best_id)
        end

        return node_roles
    end

    function plot_bionetwork_3d(net::BioNetwork, node_roles::Dict{Symbol, Int})
        role_color = Dict(
            :Input_A  => :blue,
            :Input_B  => :orange,
            :Output_0 => :green,
            :Output_1 => :red,
        )
        role_ids = Set(values(node_roles))

        # Arêtes en gris
        p = plot3d(legend = :outertopright,
                   title  = "Réseau cubique — topologie",
                   xlabel = "X", ylabel = "Y", zlabel = "Z")

        for (id1, id2, _) in net.edges
            id1 == id2 && continue
            xs = [net.nodes[id1].pos[1], net.nodes[id2].pos[1]]
            ys = [net.nodes[id1].pos[2], net.nodes[id2].pos[2]]
            zs = [net.nodes[id1].pos[3], net.nodes[id2].pos[3]]
            plot3d!(p, xs, ys, zs, color=:lightgray, lw=0.5, label=false)
        end

        # Nœuds ordinaires
        plain_ids = [id for id in keys(net.nodes) if id ∉ role_ids]
        if !isempty(plain_ids)
            xs = [net.nodes[id].pos[1] for id in plain_ids]
            ys = [net.nodes[id].pos[2] for id in plain_ids]
            zs = [net.nodes[id].pos[3] for id in plain_ids]
            scatter3d!(p, xs, ys, zs, color=:gray, ms=4, label="interneurone")
        end

        # Nœuds rôles — un scatter par rôle pour la légende
        for (role, id) in node_roles
            x = net.nodes[id].pos[1]
            y = net.nodes[id].pos[2]
            z = net.nodes[id].pos[3]
            scatter3d!(p, [x], [y], [z],
                       color  = role_color[role],
                       ms     = 10,
                       label  = string(role))
        end

        return p
    end

    function plot_bionetwork(net::BioNetwork)
        x = [b.pos[1] for b in values(net.nodes)]
        y = [b.pos[2] for b in values(net.nodes)]
        scatter(x, y, title="Réseau de bactéries", xlabel="X", ylabel="Y", legend=false)
        
        for (id1, id2, dist) in net.edges
            if id1 != id2
                x_coords = [net.nodes[id1].pos[1], net.nodes[id2].pos[1]]
                y_coords = [net.nodes[id1].pos[2], net.nodes[id2].pos[2]]
                plot!(x_coords, y_coords, color=:gray, lw=0.5)
            end
        end
        
        display(current())
        
    end
end
