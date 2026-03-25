module BioNetworks #
    using ..Bacterias
    using Catalyst
    using LinearAlgebra 

    export BioNetwork, add_bacterium!, build_edges!

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
end
