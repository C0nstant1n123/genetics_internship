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
    create_hebbian_model,
    create_hebbian_stable_model,
    create_hebbian_non_spike_model,
    create_ssa_template,
    notify_bacterium!,
    SpikeSchedule,
    make_spike_schedule,
    spike!







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
        @parameters t n v_max_D v_max_M v_max_M2 v_max_I v_max_C v_max_T v_max_E_ext v_max_inhib v_max_inhib_2 K_MD K_ID K_DM K_IM K_EeM K_EiM K_MI K_MC K_MT K_MEe K_IC k_transl k_deg_mRNA k_deg_D k_deg_M k_deg_I k_deg_C k_deg_E_ext k_deg_E_int v_max_diff_D v_max_diff_E_ext v_max_diff_C  K_D_diff k_echo_int
        @species D(t) M(t) I(t) C(t) E_int(t) T(t) E_ext(t) D_diff(t) C_diff(t) E_ext_diff(t) mRNA_D(t) mRNA_M(t) mRNA_I(t) mRNA_C(t) mRNA_T(t) mRNA_E_ext(t)
        rxs = [
            #Transcription
            Reaction(v_max_D*(M^n/(K_MD^n +M^n))*(K_ID^n/(K_ID^n+I^n))*(K_OD^n/(K_OD^n+O^n)),nothing,[mRNA_D]),
            Reaction((v_max_M*((D^n/(K_DM^n+D^n))*(K_IM^n/(K_IM^n+I^n)))+v_max_M2*((E_ext^n/(K_EeM^n+E_ext^n))*(E_int^n/(K_EiM^n+E_int^n)))),nothing,[mRNA_M]),
            Reaction(v_max_I*(M^n/(K_MI^n+M^n)),nothing,[mRNA_I]),
            Reaction(v_max_C*(M^n/(K_MC^n+M^n)),nothing,[mRNA_C]),
            Reaction(v_max_T*(M^n/(K_MT^n+M^n)),nothing,[mRNA_T]),
            Reaction(v_max_E_ext*(M^n/(K_MEe^n+M^n)),nothing,[mRNA_E_ext]),
            Reaction(v_max_O*((M^n/(K_MO^n+M^n))),nothing,[mRNA_O]),
            #Echo and inhibition Processes
            Reaction(k_echo_int, [T],[E_int]),
            Reaction(v_max_inhib*(C^n/(C^n+K_IC^n)),[C,M],nothing), #[C]
            Reaction(v_max_inhib_2*((I^n/(K_IM^n+I^n))),[I,M],[I]),
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
            Reaction(k_deg_mRNA,[mRNA_O],nothing),
            Reaction(k_deg_D,[D],nothing),
            Reaction(k_deg_M,[M],nothing),
            Reaction(k_deg_I,[I],nothing),
            Reaction(k_deg_C,[C],nothing),
            Reaction(k_deg_E_ext,[E_ext],nothing),
            Reaction(k_deg_E_int,[E_int],nothing),
            Reaction(k_deg_O,[O],nothing),
            #Secretion
            Reaction(v_max_diff_D*(D^n/(K_D_diff^n+D^n)),[D],[D_diff]),
            Reaction(v_max_diff_E_ext,[E_ext],[E_ext_diff]),
            Reaction(v_max_diff_C,[C],[C_diff]),
            Reaction(v_max_diff_O,[O],[O_diff])
        ]
        rs = ReactionSystem(rxs, t, [D, M, I, C, E_int, T, E_ext, D_diff, C_diff, E_ext_diff, mRNA_D, mRNA_M, mRNA_I, mRNA_C, mRNA_T, mRNA_E_ext],[n,v_max_D ,v_max_M ,v_max_M2 ,v_max_I ,v_max_C ,v_max_T ,v_max_E_ext ,v_max_inhib, v_max_inhib_2, K_MD,K_ID,K_DM,K_IM,K_EeM,K_EiM,K_MI,K_MC,K_MT,K_MEe,K_IC,k_transl,k_deg_mRNA,k_deg_D,k_deg_M,k_deg_I,k_deg_C,k_deg_E_ext,k_deg_E_int,v_max_diff_D ,v_max_diff_E_ext ,v_max_diff_C,K_D_diff,k_echo_int]; name=name)
        p_defaults = [
            # --- Calibration 2026-03 (défaut) ---
            8.0,    # n
            3.0,    # v_max_D
            0.5,    # v_max_M
            0.5,    # v_max_M2
            0.01,    # v_max_I
            0.3,    # v_max_C
            0.5,    # v_max_T
            0.5,    # v_max_E_ext
            0.065,   # v_max_inhib
            1.0,   #v_max_inhib_2
            100.0,  # K_MD
            100.0,  # K_ID
            50.0,   # K_DM
            50.0,   # K_IM
            1.0,   # K_EeM
            20.0,   # K_EiM
            100.0,   # K_MI
            100.0,  # K_MC
            100.0,  # K_MT
            100.0,  # K_MEe
            1.0,   # K_IC
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
            1.0,    # k_echo_int

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

    function create_hebbian_stable_model(name::Symbol)
    # 1. Déclaration de tous les paramètres
    @parameters t n m v_max_D v_max_M v_max_learn v_max_I v_max_C v_max_T v_max_E_ext v_max_CS v_max_ES v_max_inhib K_MD K_DM K_IM K_SM K_MI K_MC K_MT K_MEe K_CS K_EiS K_EiS_2 K_EeS K_act_S K_inh_S k_transl k_transl_M k_transl_S k_deg_mRNA k_deg_mRNA_M k_deg_mRNA_S k_deg_D k_deg_M k_deg_I k_deg_C k_deg_T k_deg_E_ext k_deg_E_int k_deg_S v_max_diff_D v_max_diff_E_ext v_max_diff_C K_D_diff k_echo_int v_max_O K_MO K_OD K_IS K_DS k_deg_O v_max_diff_O v_ground

    # 2. Déclaration de toutes les espèces
    @species D(t) M(t) I(t) C(t) E_int(t) T(t) E_ext(t) S(t) O(t) D_diff(t) C_diff(t) E_ext_diff(t) O_diff(t) mRNA_D(t) mRNA_M(t) mRNA_I(t) mRNA_C(t) mRNA_T(t) mRNA_E_ext(t) mRNA_S(t) mRNA_O(t)
    
    rxs = [
        # --- Création mRNA ---
        Reaction(v_max_D*(M^n/(K_MD^n +M^n)*(K_IM^n/(K_IM^n+I^n))*(K_OD^n/(K_OD^n+O^n))), nothing, [mRNA_D]),
        Reaction(v_max_M * (D^m / (K_DM^m + D^m)) + v_max_learn * ( S^m / (K_act_S^m + S^m + (S^(2*m) / K_inh_S^(2*m))) ) + v_ground, nothing, [mRNA_M]),
        Reaction(v_max_I*(M^n/(K_MI^n+M^n)), nothing, [mRNA_I]),
        Reaction(v_max_C*(M^n/(K_MC^n+M^n)), nothing, [mRNA_C]),
        Reaction(v_max_T*(M^n/(K_MT^n+M^n)), nothing, [mRNA_E_int]),
        Reaction(v_max_E_ext*(M^n/(K_MEe^n+M^n)), nothing, [mRNA_E_ext]),
        Reaction(v_max_O*(M^n/(K_MO^n+M^n)), nothing, [mRNA_O]),
        
        # --- Apprentissage (LTP) et Oubli (LTD) ---
        # Dégradation active pénalisée par C (Oubli direct de la protéine S)
        Reaction(v_max_CS*((C^n/(K_CS^n+C^n))*(K_DS^n/(D^n+K_DS^n))*(K_IS^n/(K_IS^n+I^n))), [S], nothing),
        # Apprentissage par coïncidence temporelle E_int et E_ext (Création de mRNA_S)
        Reaction(v_max_ES * (E_int^n/(K_EiS^n+E_int^n)) * (E_ext^n/(K_EeS^n+E_ext^n)) * (K_DS^n/(D^n+K_DS^n) * (K_IS^n/(K_IS^n+I^n))), nothing, [mRNA_S]),
        
        # --- Inhibition / Delay Processes ---
        Reaction(v_max_inhib, [I, D], [I]),
        
        # --- Translation ---
        Reaction(k_transl, [mRNA_D], [mRNA_D, D]),
        Reaction(k_transl_M, [mRNA_M], [mRNA_M, M]),
        Reaction(k_transl_M, [mRNA_I], [mRNA_I, I]),
        Reaction(k_transl, [mRNA_C], [mRNA_C, C]),
        Reaction(k_transl, [mRNA_E_int], [mRNA_E_int, E_int]),
        Reaction(k_transl, [mRNA_E_ext], [mRNA_E_ext, E_ext]),
        Reaction(k_transl_S, [mRNA_S], [mRNA_S, S]),
        Reaction(k_transl, [mRNA_O], [mRNA_O, O]),
        
        
        # --- Degradation ---
        Reaction(k_deg_mRNA, [mRNA_D], nothing),
        Reaction(k_deg_mRNA_M, [mRNA_M], nothing),
        Reaction(k_deg_mRNA, [mRNA_I], nothing),
        Reaction(k_deg_mRNA, [mRNA_C], nothing),
        Reaction(k_deg_mRNA, [mRNA_T], nothing),
        Reaction(k_deg_mRNA, [mRNA_E_ext], nothing),
        Reaction(k_deg_mRNA_S, [mRNA_S], nothing),
        Reaction(k_deg_mRNA, [mRNA_O], nothing),

        Reaction(k_deg_D, [D], nothing),
        Reaction(k_deg_M, [M], nothing),
        Reaction(k_deg_I, [I], nothing),
        Reaction(k_deg_C, [C], nothing),
        Reaction(k_deg_T, [T], nothing), # Ajouté pour le k_deg_T=0.5
        Reaction(k_deg_E_ext, [E_ext], nothing),
        Reaction(k_deg_E_int, [E_int], nothing),
        Reaction(k_deg_S, [S], nothing),
        Reaction(k_deg_O,[O],nothing),
        
        # --- Secretion ---
        Reaction(v_max_diff_D*(D^n/(K_D_diff^n+D^n)), [D], [D_diff]),
        Reaction(v_max_diff_E_ext, [E_ext], [E_ext_diff]),
        Reaction(v_max_diff_C, [C], [C_diff]),
        Reaction(v_max_diff_O,[O],[O_diff])
    ]

    # 3. Création du système avec TOUTES les espèces et paramètres
    rs = ReactionSystem(rxs, t,
        [D, M, I, C, E_int, T, E_ext, S, O, D_diff, C_diff, E_ext_diff, O_diff, mRNA_D, mRNA_M, mRNA_I, mRNA_C, mRNA_T, mRNA_E_ext, mRNA_S, mRNA_O],
        [n, m, v_max_D, v_max_M, v_max_learn, v_max_I, v_max_C, v_max_T, v_max_E_ext, v_max_CS, v_max_ES, v_max_inhib, K_MD, K_DM, K_IM, K_SM, K_MI, K_MC, K_MT, K_MEe, K_CS, K_EiS, K_EiS_2, K_EeS, K_act_S, K_inh_S, k_transl, k_transl_M, k_transl_S, k_deg_mRNA, k_deg_mRNA_M, k_deg_mRNA_S, k_deg_D, k_deg_M, k_deg_I, k_deg_C, k_deg_T, k_deg_E_ext, k_deg_E_int, k_deg_S, v_max_diff_D, v_max_diff_E_ext, v_max_diff_C, K_D_diff, k_echo_int, v_max_O, K_MO, K_OD, K_IS, K_DS, k_deg_O, v_max_diff_O, v_ground];
        name=name)

    # 4. Paramètres par défaut
    p_defaults = [
        # Paramètres structurels
        20.0,   # n
        1.0,    # m (coefficient de Hill pour l'apprentissage S)
        # Vmax de transcription
        10.0,   # v_max_D
        1.0,    # v_max_M
        0.096,    # v_max_learn (Force avec laquelle la mémoire S réactive M)
        2.0,    # v_max_I
        0.5,    # v_max_C
        0.8,    # v_max_T
        1.0,    # v_max_E_ext
        0.5,    # v_max_CS (Force de l'oubli actif par C)
        0.5,    # v_max_ES (Force de l'apprentissage par coïncidence)
        0.0,    # v_max_inhib
        # Constantes de Michaelis (K)
        100.0,  # K_MD
        80.0,   # K_DM
        20.0,   # K_IM
        50.0,   # K_SM (Seuil d'activation de la mémoire)
        100.0,  # K_MI
        100.0,  # K_MC
        100.0,  # K_MT
        100.0,  # K_MEe
        3.0,   # K_CS
        3.0,   # K_EiS
        80.0,  # K_EiS_2
        1.0,   # K_EeS
        25.0,  # K_act_S
        100.0, # K_inh_S
        # Cinétiques générales
        1.5,    # k_transl
        1.5,    # k_transl_M
        0.5,    # k_transl_S (très lent : la mémoire se construit progressivement)
        0.23,   # k_deg_mRNA
        0.23,   # k_deg_mRNA_M
        0.23,   # k_deg_mRNA_S
        0.1,  # k_deg_D
        7e-3,   # k_deg_M
        0.1,  # k_deg_I
        0.02,   # k_deg_C
        0.5,    # k_deg_T (Modèle stable : T plus instable)
        0.9,    # k_deg_E_ext
        0.02,   # k_deg_E_int
        1e-7,   # k_deg_S (Très lent : la mémoire persiste longtemps)
        # Diffusion et communication
        1.0,    # v_max_diff_D
        0.5,    # v_max_diff_E_ext
        0.1,    # v_max_diff_C
        100.0,  # K_D_diff
        1.0,    # k_echo_int (Modèle stable : écho réduit)
        # --- Espèce O (inhibiteur diffusible) ---
        0.8,    # v_max_O (= v_max_T)
        120.0,  # K_MO
        1.0,    # K_OD
        1.0,    # K_IS
        50.0,   # K_DS
        0.02,   # k_deg_O (= k_deg_E_int)
        1.0,    # v_max_diff_O (= v_max_diff_D)
        0.0,    # v_ground (taux basal de production de mRNA_M — activité de fond)
    ]

    return complete(rs), p_defaults
end

# ==============================================================================
# create_hebbian_non_spike_model
# Modèle ODE sans spikes intracellulaires.
# Chaque bactérie apprend uniquement de ses voisines (pas d'elle-même) :
#   - D, E_ext, C, O reçus par diffusion depuis les voisines → intracellulaire
#   - M intègre D reçu + mémoire S
#   - Spike de M (M > K_spike)      → émission directe D_diff, E_ext_diff, C_diff, T
#   - Spike fort  (M > K_spike_O)   → émission D_diff O_diff en plus
# ==============================================================================
function create_hebbian_non_spike_model(name::Symbol)

    @parameters t n m l v_max_M v_max_learn v_ground v_max_CS v_max_ES K_DM K_SM K_CS K_EiS K_EeS K_act_S K_inh_S K_IS K_DS K_spike K_spike_O v_max_spike_D v_max_spike_E v_max_spike_C v_max_spike_T v_max_spike_O k_transl_M k_transl_S k_deg_mRNA_M k_deg_mRNA_S k_deg_M k_deg_E_int k_deg_E_ext k_deg_C k_deg_T k_deg_S k_deg_D k_deg_O k_echo_int v_max_diff_D v_max_diff_E v_max_diff_C v_max_diff_O K_D_diff K_OD

    @species D(t) M(t) I(t) E_int(t) T(t) E_ext(t) C(t) O(t) S(t) D_diff(t) E_ext_diff(t) C_diff(t) O_diff(t) mRNA_M(t) mRNA_S(t)

    rxs = [
        # --- Intégration : D reçu (voisines) + mémoire S + basal → mRNA_M ---
        Reaction(v_max_M * (D^l / (K_DM^l + D^l)) * (K_OD^n / (K_OD^n + O^n)) + v_max_learn * 1.0 * (S^m / (K_act_S^m + S^m + S^(2*m) / K_inh_S^(2*m))) *  (K_OD^n / (K_OD^n + O^n)) + v_ground, nothing, [mRNA_M]), #0.33

        # --- Apprentissage : coïncidence E_int × E_ext → mRNA_S ---
        Reaction(v_max_ES * (E_int^n/(K_EiS^n+E_int^n)) * (E_ext^n/(K_EeS^n+E_ext^n)) , nothing, [mRNA_S]),

        # --- Oubli : C × (pas D) × (pas I) → dégrade S ---
        Reaction(v_max_CS * (C^n/(K_CS^n+C^n)), [S], nothing),#; only_use_rate=true), * (S^4/(0.5^4 + S^4))

        # --- Délai interne : T → E_int ---
        Reaction(k_echo_int, [T], [E_int]),

        # --- Translation ---
        Reaction(k_transl_M, [mRNA_M], [mRNA_M, M]),
        Reaction(k_transl_S, [mRNA_S], [mRNA_S, S]),

        # --- Dégradation mRNA ---
        Reaction(k_deg_mRNA_M, [mRNA_M], nothing),
        Reaction(k_deg_mRNA_S, [mRNA_S], nothing),

        # --- Dégradation protéines internes ---
        Reaction(k_deg_M,     [M],     nothing),
        Reaction(k_deg_E_int, [E_int], nothing),
        Reaction(k_deg_E_ext, [E_ext], nothing),
        Reaction(k_deg_C,     [C],     nothing),
        Reaction(k_deg_T,     [T],     nothing),
        Reaction(k_deg_S,     [S],     nothing),
        Reaction(k_deg_D,     [D],     nothing),
        Reaction(k_deg_O,     [O],     nothing),

        # --- Spike faible (M > K_spike) → émission directe dans le milieu ---
        Reaction(v_max_spike_D * (M^n/(K_spike^n+M^n)), nothing, [D_diff]),
        Reaction(v_max_spike_E * (M^n/(K_spike^n+M^n)), nothing, [E_ext_diff]),
        Reaction(v_max_spike_C * (M^n/(K_spike^n+M^n)), nothing, [C_diff]),
        Reaction(v_max_spike_T * (M^n/(K_spike^n+M^n)), nothing, [E_int]),

        # --- Spike fort (M > K_spike_O) → émission O_diff ---
        Reaction(v_max_spike_O * (M^n/(K_spike_O^n+M^n)), nothing, [O_diff]),

        # --- Dégradation des molécules diffusibles ---
        Reaction(v_max_diff_D * (D_diff^n/(K_D_diff^n+D_diff^n)), [D_diff], nothing),
        Reaction(v_max_diff_E, [E_ext_diff], nothing),
        Reaction(v_max_diff_C, [C_diff],     nothing),
        Reaction(v_max_diff_O, [O_diff],     nothing),
    ]

    rs = ReactionSystem(rxs, t,
        [D, M, I, E_int, T, E_ext, C, O, S, D_diff, E_ext_diff, C_diff, O_diff, mRNA_M, mRNA_S],
        [n, m, l, v_max_M, v_max_learn, v_ground, v_max_CS, v_max_ES, K_DM, K_SM, K_CS, K_EiS, K_EeS, K_act_S, K_inh_S, K_IS, K_DS, K_spike, K_spike_O, v_max_spike_D, v_max_spike_E, v_max_spike_C, v_max_spike_T, v_max_spike_O, k_transl_M, k_transl_S, k_deg_mRNA_M, k_deg_mRNA_S, k_deg_M, k_deg_E_int, k_deg_E_ext, k_deg_C, k_deg_T, k_deg_S, k_deg_D, k_deg_O, k_echo_int, v_max_diff_D, v_max_diff_E, v_max_diff_C, v_max_diff_O, K_D_diff, K_OD];
        name=name)

    p_defaults = [
        20.0,   # n
        1.0,    # m
        1.0,    # l
        0.4,    # v_max_M
        0.5,    # v_max_learn
        0.0,    # v_ground
        0.002,  # v_max_CS
        0.01,   # v_max_ES
        20.0,   # K_DM
        50.0,   # K_SM
        4.0,    # K_CS
        20.0,   # K_EiS
        1.0,    # K_EeS
        25.0,   # K_act_S
        100.0,  # K_inh_S
        1.0,    # K_IS
        40.0,   # K_DS
        120.0,  # K_spike
        150.0,  # K_spike_O
        1.0,    # v_max_spike_D
        0.5,    # v_max_spike_E
        0.5,    # v_max_spike_C
        0.5,    # v_max_spike_T
        0.3,    # v_max_spike_O
        0.15,   # k_transl_M
        0.5,    # k_transl_S
        0.23,   # k_deg_mRNA_M
        0.23,   # k_deg_mRNA_S
        0.005,  # k_deg_M
        0.001,  # k_deg_E_int
        0.8,    # k_deg_E_ext
        0.05,   # k_deg_C
        0.0,    # k_deg_T
        1e-7,   # k_deg_S
        0.02,   # k_deg_D
        0.02,   # k_deg_O
        0.1,    # k_echo_int
        1.0,    # v_max_diff_D
        0.5,    # v_max_diff_E
        0.4,    # v_max_diff_C
        0.5,    # v_max_diff_O
        100.0,  # K_D_diff
        1.0,    # K_OD
    ]

    return complete(rs), p_defaults
end



# ==============================================================================
# 2.0_______The obect Bacterium_________
# ==============================================================================


    # Paramétrique sur I = type concret de l'intégrateur.
    # Julia peut ainsi spécialiser step_bacterium! et notify_bacterium!
    # → élimine le dynamic dispatch sur b.integrator qui était typé Any.
    mutable struct Bacterium{I}
        id::Int
        pos::Vector{Float64}
        integrator::I
        model::ReactionSystem
        species_index::Dict{Symbol, Int}

        function _build_species_index(system::ReactionSystem)::Dict{Symbol, Int}
            idx = Dict{Symbol, Int}()
            for (i, s) in enumerate(species(system))
                s_clean = replace(string(s), "(t)" => "")
                short   = Symbol(split(s_clean, "₊")[end])
                idx[short] = i
            end
            return idx
        end

        # Constructeur original — crée le JumpProblem/ODEProblem from scratch
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
            new{typeof(integ)}(id, pos, integ, system, _build_species_index(system))
        end

        function Bacterium(id, pos, system::ReactionSystem,
                           template::JumpProcesses.JumpProblem, params_dict::Dict)
            jprob = remake(template; p=params_dict)
            integ = init(jprob, SSAStepper())
            new{typeof(integ)}(id, pos, integ, system, _build_species_index(system))
        end

        # Constructeur template — init directement depuis le JumpProblem template.
        # init() copie u0 et crée un agrégateur indépendant → pas d'état partagé.
        # Toutes les bactéries initialisées depuis le même template ont le même type I
        # → Vector{Bacterium{I}} possible → static dispatch sur step_bacterium!
        function Bacterium(id, pos, system::ReactionSystem,
                           template::JumpProcesses.JumpProblem)
            integ = init(template, SSAStepper())
            new{typeof(integ)}(id, pos, integ, system, _build_species_index(system))
        end
    end

    # Crée le JumpProblem template (appeler une seule fois par circuit+params)
    function create_ssa_template(system::ReactionSystem,
                                 params_dict, u0_dict,
                                 tspan=(0.0, 10000.0))::JumpProcesses.JumpProblem
        dprob = DiscreteProblem(system, u0_dict, tspan, params_dict)
        return JumpProblem(system, dprob, Direct(), save_positions=(false, false))
    end


# ==============================================================================
# 2.1_____ Some functions for helping run ______
# ==============================================================================

    # --------------------------------------------------------------------------
    # Spike utilities pour create_hebbian_non_spike_model
    # --------------------------------------------------------------------------

    """
        SpikeSchedule

    Planning d'un spike : indices de steps et quantité à injecter par step
    pour chaque espèce.

    Champs :
    - step_start, step_end : indices dans la boucle temporelle
    - per_step : Dict{Symbol,Float64} — quantité ajoutée à chaque step
    """
    struct SpikeSchedule
        step_start :: Int
        step_end   :: Int
        per_step   :: Dict{Symbol, Float64}
    end

    """
        make_spike_schedule(t_start, duration, dt,
                            amp_D_diff, amp_E_ext_diff, amp_C_diff, amp_T;
                            amp_O_diff=0.0) -> SpikeSchedule

    Pré-calcule un spike : convertit les amplitudes totales en quantité par step.

    Arguments :
    - t_start   : temps de début du spike (s)
    - duration  : largeur temporelle du spike (s)
    - dt        : pas de temps de la simulation (s)
    - amp_*     : amplitude totale intégrée de chaque molécule sur le spike
                  (sera distribuée uniformément sur les steps)
    - amp_O_diff: optionnel, 0 par défaut (spike fort uniquement)
    """
    function make_spike_schedule(t_start::Float64, duration::Float64, dt::Float64,
                                 amp_D_diff::Float64, amp_E_ext_diff::Float64,
                                 amp_C_diff::Float64, amp_T::Float64;
                                 amp_O_diff::Float64=0.0)::SpikeSchedule
        n_steps    = max(1, round(Int, duration / dt))
        step_start = round(Int, t_start / dt) + 1
        step_end   = step_start + n_steps - 1
        per_step   = Dict{Symbol, Float64}(
            :D_diff     => amp_D_diff     / n_steps,
            :E_ext_diff => amp_E_ext_diff / n_steps,
            :C_diff     => amp_C_diff     / n_steps,
            :T          => amp_T          / n_steps,
        )
        amp_O_diff > 0.0 && (per_step[:O_diff] = amp_O_diff / n_steps)
        return SpikeSchedule(step_start, step_end, per_step)
    end

    """
        spike!(b::Bacterium, schedule::SpikeSchedule, step::Int)

    À appeler à chaque pas de la boucle temporelle.
    Injecte les molécules du spike si `step` est dans la fenêtre du schedule.
    """
    function spike!(b::Bacterium, schedule::SpikeSchedule, step::Int)
        schedule.step_start <= step <= schedule.step_end || return
        for (sp, val) in schedule.per_step
            idx = get(b.species_index, sp, 0)
            idx == 0 && continue
            b.integrator.u[idx] += val
        end
        notify_bacterium!(b)
    end



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
        return get(b.species_index, species_name, nothing)
    end

    function get_species(b::Bacterium, species_name::Symbol)::Float64
        idx = get(b.species_index, species_name, nothing)
        isnothing(idx) && return 0.0
        return Float64(b.integrator.u[idx])
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

    """
        notify_bacterium!(b)

    Notifie l'intégrateur SSA (reset_aggregated_jumps!) après des modifications directes
    sur `b.integrator.u`. À appeler UNE SEULE FOIS après un groupe de modifications pour
    éviter 2×n_diffusible appels à reset_aggregated_jumps! par pas de temps.
    """
    function notify_bacterium!(b::Bacterium)
        _notify_integrator!(b.integrator)
    end

   






end