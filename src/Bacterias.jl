module Bacterias


    # src/Bacterias.jl
    using Catalyst
    using JumpProcesses
    using OrdinaryDiffEq
    using Distributions
    export create_genetic_simple_circuit_input, 
    create_genetic_simple_circuit_output, 
    create_iFFL_circuit, 
    create_propagating_iFFL_circuit, 
    Bacterium, step_bacterium!, 
    get_species, add_input!, 
    map_symbols_to_species, 
    Bacterium, set_species!, 
    create_genetic_simple_circuit_input_integrated, 
    set_param!, 
    create_simple_death_birth_model, 
    create_genetic_hill_repeter_input, 
    create_genetic_hill_repeter_output, 
    create_burst_circuit,
    create_hebbian_model







# ==============================================================================
# 1._____ Some models _____
# ==============================================================================




    function create_simple_death_birth_model(name::Symbol)
        # Simple D/B model, where only one molecule is produced and degradated 

        @parameters t k_create k_deg
        @species X(t) 

        rxs = [
            Reaction(k_create,nothing,[X]),
            Reaction(k_deg, [X], nothing)

        ]

        rs = ReactionSystem(rxs, t, [X], [k_create,k_deg]; name=name)

        p_defaults = [
            0.1,
            2.9e-3
        ]

        return complete(rs), p_defaults

    end




# ==============================================================================
# 1.1_____ Positive simple ... circuit : 2 mol => 1 input, 1 output ____________
# ==============================================================================


    function create_genetic_simple_circuit_input(name::Symbol)
        # Input molecule with external drive 

        @parameters t k_act_Y k_deg_prom_act k_transc k_deg_mRNA k_transl  k_deg_Y  k_sec
        @species X(t) Y(t) Y_trans(t) Z(t) Prom(t) Prom_act(t) mRNA(t)
        
        rxs = [
            Reaction(k_act_Y,[Prom,X],[Prom_act]),
            Reaction(k_deg_prom_act, [Prom_act], [Prom, X]), 
            Reaction(k_transc, [Prom_act], [Prom_act, mRNA]),
            Reaction(k_deg_mRNA, [mRNA], nothing),
            Reaction(k_transl, [mRNA], [mRNA, Y]),
            Reaction(k_deg_Y, [Y], nothing),
            Reaction(k_sec, [Y], [Y_trans])
            
        ]

        rs = ReactionSystem(rxs, t, [X, Y, Y_trans, Z, Prom, Prom_act, mRNA], [k_act_Y, k_deg_prom_act, k_transc, k_deg_mRNA, k_transl, k_deg_Y, k_sec]; name=name)

        p_defaults = [
            1.0,
            1.0,
            0.1,
            0.23, #0.23
            0.69,
            2.9e-3,
            1.0
        ]

        return complete(rs), p_defaults
    end

    function create_genetic_simple_circuit_input_integrated(name::Symbol)
        # Input with internal drive (Death/Birth model as a drive)

        @parameters t k_act_Y k_deg_prom_act k_transc k_deg_mRNA k_transl  k_deg_Y k_create_X k_deg_X k_sec
        @species X(t) Y(t) Y_trans(t) Z(t) Prom(t) Prom_act(t) mRNA(t)
        
        rxs = [
            Reaction(k_act_Y,[Prom,X],[Prom_act]),
            Reaction(k_deg_prom_act, [Prom_act], [Prom, X]), 
            Reaction(k_transc, [Prom_act], [Prom_act, mRNA]),
            Reaction(k_deg_mRNA, [mRNA], nothing),
            Reaction(k_transl, [mRNA], [mRNA, Y]),
            Reaction(k_create_X, nothing, [X]),
            Reaction(k_deg_X, [X], nothing),
            Reaction(k_deg_Y, [Y], nothing),
            Reaction(k_sec, [Y], [Y_trans])
            
        ]

    

        rs = ReactionSystem(rxs, t, [X, Y, Y_trans, Z, Prom, Prom_act, mRNA], [k_act_Y, k_deg_prom_act, k_transc, k_deg_mRNA, k_transl, k_deg_Y, k_create_X, k_deg_X, k_sec]; name=name)

        p_defaults = [
            0.1,
            1.0,
            0.1,
            0.23, #0.23
            0.69,
            2.9e-3,
            0.1,
            2.9e-3,
            1.0

        ]

        return complete(rs), p_defaults
    end

    function create_genetic_simple_circuit_output(name::Symbol)
        # Output 

        @parameters t k_act_Z k_deg_prom_act k_transc k_deg_mRNA k_transl k_deg_X k_deg_Y k_int  
        @species X(t) Y(t) Y_trans(t) Z(t) Prom(t) Prom_act(t) mRNA(t)
        
        rxs = [
            Reaction(k_act_Z,[Prom,Y],[Prom_act]),
            Reaction(k_deg_prom_act, [Prom_act], [Prom, Z]), 
            Reaction(k_transc, [Prom_act], [Prom_act, mRNA]),
            Reaction(k_deg_mRNA, [mRNA], nothing),
            Reaction(k_transl, [mRNA], [mRNA, Y]),
            Reaction(k_deg_X, [Y], nothing),
            Reaction(k_deg_Y, [Z], nothing),
            Reaction(k_int, [Y_trans], [Y])
            
        ]

        rs = ReactionSystem(rxs, t, [X, Y, Y_trans, Z, Prom, Prom_act, mRNA], [k_act_Z, k_deg_prom_act, k_transc, k_deg_mRNA, k_transl, k_deg_X, k_deg_Y, k_int]; name=name)
        p_defaults = [
            1.0,
            1.0,
            0.1,
            0.23,
            0.69,
            2.9e-3,
            2.9e-3,
            1000
        ]

        return complete(rs), p_defaults
    end
           


# ==============================================================================
# 1.2 _____ Positive simple regulatory circuit : Add Hill model 
# ==============================================================================


    function create_genetic_hill_repeter_input(name::Symbol)
        # Internal input : Death/Birth model

        @parameters t v_max K_input n k_transl k_deg_mRNA k_deg_X k_deg_Y k_sec k_create_X
        @species X(t) Y(t) Y_trans(t) Z(t) mRNA(t)

        rxs = [
            Reaction(v_max*(X^n/(K_input^n+X^n)),nothing,[mRNA]),
            Reaction(k_transl,[mRNA],[mRNA,Y]),
            Reaction(k_deg_mRNA, [mRNA], nothing),
            Reaction(k_sec,[Y],[Y_trans]),
            Reaction(k_deg_X,[X],nothing),
            Reaction(k_deg_Y,[Y],nothing),
            Reaction(k_create_X, nothing,[X])
        ]

        rs = ReactionSystem(rxs, t, [X, Y, Y_trans, Z, mRNA], [v_max, K_input, n, k_transl, k_deg_mRNA, k_deg_X, k_deg_Y, k_sec, k_create_X]; name=name)

        p_defaults = [
            1.0,
            100.0,
            12.0,
            5.0,
            0.23,
            9e-5,
            2.9e-3,
            0.0,  #1.0
            9e-3
        ]

        return complete(rs), p_defaults

    end


    function create_genetic_hill_repeter_output(name::Symbol)
        # Output

        @parameters t v_max K_output n k_transl k_deg_mRNA k_deg_Y k_deg_Z k_int
        @species X(t) Y(t) Y_trans(t) Z(t) mRNA(t)

        rxs = [
            Reaction(v_max*(Y^n/(K_output^n+Y^n)),nothing,[mRNA]),
            Reaction(k_transl,[mRNA],[mRNA,Z]),
            Reaction(k_deg_mRNA, [mRNA], nothing),
            Reaction(k_int,[Y_trans],[Y]),
            Reaction(k_deg_Z,[Z],nothing),
            Reaction(k_deg_Y,[Y],nothing)
        ]

        rs = ReactionSystem(rxs, t, [X, Y, Y_trans, Z, mRNA], [v_max, K_output, n, k_transl, k_deg_mRNA, k_deg_Z, k_deg_Y, k_int]; name=name)

        p_defaults = [
            9e-3,
            500.0,
            1.0,
            5.0,
            0.23,
            9e-5,
            2.9e-3,
            100.0
        ]

        return complete(rs), p_defaults

    end

            

# ==============================================================================
# 1.3_______ Simple IFFL model _________
# ==============================================================================


    function create_iFFL_circuit(name::Symbol)
        @parameters t k_prod_y k_prod_z k_inh k_rel k_deg_y k_deg_z
        @species A(t) B(t) C(t) Gz_on(t) Gz_off(t)

        rxs = [
            Reaction(k_prod_y, [A], [A, B]),       
            Reaction(k_prod_z, [A, Gz_on], [A, C, Gz_on]),  
            Reaction(k_inh, [B, Gz_on], [Gz_off]),  
            Reaction(k_rel, [Gz_off], [B, Gz_on]),       
            Reaction(k_deg_z, [C], nothing),
            Reaction(k_deg_y, [B], nothing)   
        ]

        rs = ReactionSystem(rxs, t, [A, B, C, Gz_on, Gz_off], [k_prod_y, k_prod_z, k_inh, k_rel, k_deg_y, k_deg_z]; name=name)

        p_defaults = [10.0, 50.0, 10.0, 5.0, 0.05,0.2]

        return complete(rs), p_defaults
    end


    function create_propagating_iFFL_circuit(name::Symbol)
        @parameters t k_prod_y k_prod_z k_inh k_rel k_deg k_z_x
        @species A(t) B(t) C(t) Gz_on(t) Gz_off(t)
        rxs = [
            Reaction(k_z_x, [c], [A, C]),
            Reaction(k_prod_y, [A], [A, B]),       
            Reaction(k_prod_z, [A, Gz_on], [A, C, Gz_on]),  
            Reaction(k_inh, [B, Gz_on], [Gz_off]),  
            Reaction(k_rel, [Gz_off], [B, Gz_on]),       
            Reaction(k_deg, [C], nothing),
            Reaction(k_deg, [B], nothing)   
        ]
        rs = ReactionSystem(rxs, t, [A, B, C, Gz_on, Gz_off], [k_prod_y, k_prod_z, k_inh, k_rel, k_deg, k_z_x]; name=name)


        p_defaults = [5.0, 5.0, 5.0, 1.0, 0.1, 5.0]

        return complete(rs), p_defaults
    end



# ==============================================================================
# 1.4 _________First try of impleting a plastic gentic circuit : model with bursts__________
# ==============================================================================


    function create_burst_circuit(name::Symbol)
        @parameters t n v_max_X v_max_Y v_max_Z v_max_diff K_ZX K_YX K_XY K_YZ k_transl k_deg_X k_deg_Y k_deg_Z k_deg_mRNA K_X_diff
        @species X(t) Y(t) Z(t) mRNA_X(t) mRNA_Y(t) mRNA_Z(t) X_diff(t)

        rxs = [
            # Transcription
            Reaction(v_max_X*(K_ZX^n/(K_ZX^n+Z^n))*(Y^n/(K_YX^n+Y^n)),nothing,[mRNA_X]),
            Reaction(v_max_Y*(X^n/(K_XY^n+X^n)),nothing,[mRNA_Y]),
            Reaction(v_max_Z*(Y^n/(K_YZ^n+Y^n)),nothing,[mRNA_Z]),
            # Translation
            Reaction(k_transl,[mRNA_X],[mRNA_X,X]),
            Reaction(k_transl,[mRNA_Y],[mRNA_Y,Y]),
            Reaction(k_transl,[mRNA_Z],[mRNA_Z,Z]),
            # Degradation
            Reaction(k_deg_X,[X],nothing),
            Reaction(k_deg_Y,[Y],nothing),
            Reaction(k_deg_Z,[Z],nothing),
            Reaction(k_deg_mRNA,[mRNA_X],nothing),
            Reaction(k_deg_mRNA,[mRNA_Y],nothing),
            Reaction(k_deg_mRNA,[mRNA_Z],nothing),
            # Secretion
            Reaction(v_max_diff*(X^n/(K_X_diff^n+X^n)),[X],[X_diff])
            ]

        rs = ReactionSystem(rxs, t, [X, Y, Z, mRNA_X, mRNA_Y, mRNA_Z, X_diff], [n, v_max_X, v_max_Y, v_max_Z, v_max_diff, K_ZX, K_YX, K_XY, K_YZ, k_transl, k_deg_X, k_deg_Y, k_deg_Z, k_deg_mRNA, K_X_diff]; name=name)

        p_defaults = [
            # --- Calibration 2026-03 : timescales biologiques réalistes ---
            # Unité de temps = 1 minute (ancré sur k_deg_mRNA = 0.23 → t½ ≈ 3 min)
            #
            # X_ss ≈ 164 mol  (v_max=3.0, k_eff=0.119)
            # Y_ss ≈ 283 mol  (v_max=2.0, k_deg=0.046)
            # Z_ss ≈ 362 mol  (v_max=0.5, k_deg=0.009)
            # → Burst ≈ 32 min, période réfractaire ≈ 57 min
            8.0,    # n         — Hill (était 20, max biologique ~4)
            3.0,    # v_max_X   — inchangé
            2.0,    # v_max_Y   — était 0.05 (Y_ss était ~1630, inatteignable)
            0.5,    # v_max_Z   — était 0.05 (Z_ss était ~1090, trop lent)
            1.0,   # v_max_diff — était 0.5 (drainait X trop vite)
            90.0,   # K_ZX      — 0.25 × Z_ss (était 20, trop sensible au bruit)
            65.0,  # K_YX      — 0.35 × Y_ss (était 200, souvent inaccessible)
            40.0,   # K_XY      — 0.25 × X_ss (était 60)
            150.0,  # K_YZ      — 0.55 × Y_ss (était 500, Z ne s'activait jamais)
            1.5,    # k_transl  — inchangé
            0.069,  # k_deg_X   — t½ = 10 min  (était 0.005, t½ = 139 min)
            0.02,  # k_deg_Y   — t½ = 15 min  (était 2e-4, t½ = 3470 min)
            0.009,  # k_deg_Z   — t½ = 77 min  (était 3e-4, t½ = 2300 min)
            0.23,   # k_deg_mRNA — inchangé (ancre biologique)
            60.0,   # K_X_diff  — 0.4 × X_ss (était 200, jamais atteint)
        ]

        return(complete(rs), p_defaults)

    end


    


# ==============================================================================
# 1.5________First try of a learning hebbian model_____________________
# ==============================================================================


    function create_hebbian_model(name::Symbol)
        @parameters t n v_max_D v_max_M v_max_M2 v_max_I v_max_C v_max_T v_max_E_ext v_max_inhib K_MD K_ID K_DM K_IM K_EeM K_EiM K_MI K_MC K_MT K_MEe K_IC k_transl k_deg_mRNA k_deg_D k_deg_M k_deg_I k_deg_C k_deg_E_ext k_deg_E_int v_max_diff_D v_max_diff_E_ext v_max_diff_C  K_D_diff k_echo_int
        @species D(t) M(t) I(t) C(t) E_int(t) T(t) E_ext(t) D_diff(t) C_diff(t) E_ext_diff(t) mRNA_D(t) mRNA_M(t) mRNA_I(t) mRNA_C(t) mRNA_T(t) mRNA_E_ext(t)
        rxs = [
            #Transcription
            Reaction(v_max_D*(M^n/(K_MD^n +M^n))*(K_ID^n/(K_ID^n+I^n)),nothing,[mRNA_D]),
            Reaction((v_max_M*((D^n/(K_DM^n+D^n))*(K_IM^n/(K_IM^n+I^n)))+v_max_M2*((E_ext^n/(K_EeM^n+E_ext^n))*(E_int^n/(K_EiM^n+E_int^n)))),nothing,[mRNA_M]),
            Reaction(v_max_I*(M^n/(K_MI^n+M^n)),nothing,[mRNA_I]),
            Reaction(v_max_C*(I^n/(K_MC^n+I^n)),nothing,[mRNA_C]),
            Reaction(v_max_T*(M^n/(K_MT^n+M^n)),nothing,[mRNA_T]),
            Reaction(v_max_E_ext*(M^n/(K_MEe^n+M^n)),nothing,[mRNA_E_ext]),
            #Echo and inhibition Processes
            Reaction(k_echo_int, [T],[E_int]),
            Reaction(v_max_inhib*(K_IC^n/(I^n+K_IC^n)),[C,M],nothing), #[C]
            #Translation
            Reaction(k_transl,[mRNA_D],[mRNA_D,D]),
            Reaction(k_transl,[mRNA_M],[mRNA_M,M]),
            Reaction(k_transl,[mRNA_I],[mRNA_I,I]),
            Reaction(k_transl,[mRNA_C],[mRNA_C,C]),
            Reaction(k_transl,[mRNA_T],[mRNA_T,T]),
            Reaction(k_transl,[mRNA_E_ext],[mRNA_E_ext,E_ext]),
            #Degradation
            Reaction(k_deg_mRNA,[mRNA_D],nothing),
            Reaction(k_deg_mRNA,[mRNA_M],nothing),
            Reaction(k_deg_mRNA,[mRNA_I],nothing),
            Reaction(k_deg_mRNA,[mRNA_C],nothing),
            Reaction(k_deg_mRNA,[mRNA_T],nothing),
            Reaction(k_deg_mRNA,[mRNA_E_ext],nothing),
            Reaction(k_deg_D,[D],nothing),
            Reaction(k_deg_M,[M],nothing),
            Reaction(k_deg_I,[I],nothing),
            Reaction(k_deg_C,[C],nothing),
            Reaction(k_deg_E_ext,[E_ext],nothing),
            Reaction(k_deg_E_int,[E_int],nothing),
            #Secretion
            Reaction(v_max_diff_D*(D^n/(K_D_diff^n+D^n)),[D],[D_diff]),
            Reaction(v_max_diff_E_ext,[E_ext],[E_ext_diff]),
            Reaction(v_max_diff_C,[C],[C_diff])
        ]
        rs = ReactionSystem(rxs, t, [D, M, I, C, E_int, T, E_ext, D_diff, C_diff, E_ext_diff, mRNA_D, mRNA_M, mRNA_I, mRNA_C, mRNA_T, mRNA_E_ext],[n,v_max_D ,v_max_M ,v_max_M2 ,v_max_I ,v_max_C ,v_max_T ,v_max_E_ext ,v_max_inhib ,K_MD,K_ID,K_DM,K_IM,K_EeM,K_EiM,K_MI,K_MC,K_MT,K_MEe,K_IC,k_transl,k_deg_mRNA,k_deg_D,k_deg_M,k_deg_I,k_deg_C,k_deg_E_ext,k_deg_E_int,v_max_diff_D ,v_max_diff_E_ext ,v_max_diff_C,K_D_diff,k_echo_int]; name=name)
        p_defaults = [
            # --- Calibration 2026-03 (défaut) ---
            8.0,    # n
            3.0,    # v_max_D
            0.5,    # v_max_M
            0.5,    # v_max_M2
            0.5,    # v_max_I
            0.3,    # v_max_C
            0.5,    # v_max_T
            0.5,    # v_max_E_ext
            5e-3,   # v_max_inhib
            100.0,  # K_MD
            100.0,  # K_ID
            50.0,   # K_DM
            50.0,   # K_IM
            15.0,   # K_EeM
            50.0,   # K_EiM
            80.0,   # K_MI
            100.0,  # K_MC
            100.0,  # K_MT
            100.0,  # K_MEe
            25.0,   # K_IC
            1.5,    # k_transl
            0.23,   # k_deg_mRNA
            0.069,  # k_deg_D
            1e-3,   # k_deg_M
            0.069,  # k_deg_I
            0.02,   # k_deg_C
            0.9,    # k_deg_E_ext
            0.02,   # k_deg_E_int
            1.0,    # v_max_diff_D
            0.5,    # v_max_diff_E_ext
            0.1,    # v_max_diff_C
            100.0,  # K_D_diff
            0.1,    # k_echo_int

            # --- Calibration 2026-04 : issu de CMA-ES eval#49, loss=0.639 (meilleur observé) ---
            # Différences clés vs calibration 2026-03 :
            #   - n beaucoup plus élevé (18.5 vs 8) → switch Hill plus abrupt
            #   - k_transl très faible (0.509 vs 1.5) → production lente
            #   - k_deg_M légèrement plus rapide (4.4e-3 vs 1e-3) → mémoire moins saturante
            #   - k_deg_I très lent (1.4e-3 vs 0.069) → inhibiteur persistant
            #   - v_max_inhib très faible (3.2e-5 vs 5e-3) → inhibition C×M quasi nulle
            #   - v_max_diff_D très faible (7e-3 vs 1.0) → diffusion D locale, pas broadcast
            #   - K_D_diff faible (0.96 vs 100) → D sécrété dès les premiers spikes
            #
            # 18.49,  # n
            # 1.74,   # v_max_D
            # 0.138,  # v_max_M
            # 0.202,  # v_max_I
            # 1.430,  # v_max_C
            # 0.199,  # v_max_T
            # 0.089,  # v_max_E_ext
            # 3.17e-5,# v_max_inhib
            # 10.0,   # K_MD
            # 7.24,   # K_ID
            # 26.46,  # K_DM
            # 82.53,  # K_IM
            # 3.44,   # K_EeM
            # 13.79,  # K_EiM
            # 238.82, # K_MI
            # 31.53,  # K_MC
            # 13.99,  # K_MT
            # 298.68, # K_MEe
            # 133.92, # K_IC
            # 0.509,  # k_transl
            # 0.287,  # k_deg_mRNA
            # 0.287,  # k_deg_D
            # 4.38e-3,# k_deg_M
            # 1.39e-3,# k_deg_I
            # 1.10e-3,# k_deg_C
            # 0.500,  # k_deg_E_ext
            # 0.354,  # k_deg_E_int
            # 6.97e-3,# v_max_diff_D
            # 0.493,  # v_max_diff_E_ext
            # 4.09e-3,# v_max_diff_C
            # 0.961,  # K_D_diff
            # 16.80,  # k_echo_int
        ]
        return(complete(rs),p_defaults)
    
    end

    
# ==============================================================================
# 2.0_______The obect Bacterium_________
# ==============================================================================


    mutable struct Bacterium
        id::Int
        pos::Vector{Float64}
        integrator::Any       
        model::ReactionSystem 


        function Bacterium(id, pos, system, param_values, u0_values; mode=:ode)
            tspan = (0.0, 10000.0)
            integ = if mode == :ssa
                dprob = DiscreteProblem(system, u0_values, tspan, param_values)
                jprob = JumpProblem(system, dprob, Direct(), save_positions=(false,false))
                init(jprob, SSAStepper())
            else
                oprob = ODEProblem(system, u0_values, tspan, param_values)
                init(oprob, Tsit5())
            end
            new(id, pos, integ, system)
        end
    end


# ==============================================================================
# 2.1_____ Some functions for helping run ______
# ==============================================================================



    function set_param!(d::Dict, name::Symbol, value)
        for k in keys(d)
            if Symbol(k) == name
                d[k] = value
                return d
            end
        end
        error("Paramètre $name introuvable dans le dictionnaire")
    end

    function map_symbols_to_species(sys::ReactionSystem, u0_dict::Dict)
        new_u0 = Dict()
        
        for s in species(sys)
            str_s = string(s)
            str_clean = replace(str_s, "(t)" => "")
            short_name_str = split(str_clean, "₊")[end]
            

            sym = Symbol(short_name_str)
            
            if haskey(u0_dict, sym)
                new_u0[s] = u0_dict[sym]
            else
                println("Warning: Variable système '$short_name_str' non trouvée dans u0_dict.")
            end
        end
        
        return new_u0
    end





    function step_bacterium!(b::Bacterium, dt::Float64)
        step!(b.integrator, dt, true) 
    end

    function find_species_index(b::Bacterium, species_name::Symbol)
        target_str = string(species_name)
        
        idx = findfirst(s -> begin
            s_str = string(s)          
            s_clean = replace(s_str, "(t)" => "") 
            s_clean == target_str || endswith(s_clean, "₊" * target_str)
        end, species(b.model))

        return idx
    end




    function get_species(b::Bacterium, species_name::Symbol)
        idx = find_species_index(b, species_name)
        if isnothing(idx)
            return 0.0 
        end
        return b.integrator.u[idx]
    end



    function _notify_integrator!(integ)
        if integ isa JumpProcesses.SSAIntegrator
            reset_aggregated_jumps!(integ)
        else
            u_modified!(integ, true)
        end
    end

    function add_input!(b::Bacterium, species_name::Symbol, amount::Float64)
        idx = find_species_index(b, species_name)
        if !isnothing(idx)
            b.integrator.u[idx] += amount
            _notify_integrator!(b.integrator)
        end
    end

    function set_species!(b::Bacterium, species_name::Symbol, new_value::Float64)
        idx = find_species_index(b, species_name)
        if !isnothing(idx)
            b.integrator.u[idx] = new_value
            _notify_integrator!(b.integrator)
        end
    end

   






end