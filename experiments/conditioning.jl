# conditioning.jl — CS+/CS- associative conditioning on a 27-bacteria SSA cube.
#
# CS+ stimulus must drive the Output node; CS- must leave it silent. Plasticity
# uses the threshold-based non-spike Hebbian model (create_hebbian_non_spike_model).
#
# Usage: julia --project=. --threads=auto experiments/conditioning.jl

using Pkg, Printf, Dates
PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT)

using Catalyst, JumpProcesses, OrdinaryDiffEq, Statistics, Random, LinearAlgebra
using Plots

include(joinpath(PROJECT_ROOT, "src", "BioKan.jl"))
using .BioKan

println("=== Conditionnement CS+/CS- — non-spike model (SSA, 27 bactéries) ===\n")

# 1. Model parameters (θ₀ index → name given inline, matching parameters(circuit))
θ₀ = Float64[
    15.0,   # 1  n
    1.0,    # 2  m
    1.0,    # 3  l
    0.4,    # 4  v_max_M
    0.8,    # 5  v_max_learn
    0.0,    # 6  v_ground
    3e-5,   # 7  v_max_CS
    0.006,  # 8  v_max_ES
    20.0,   # 9  K_DM
    50.0,   # 10 K_SM
    4.0,    # 11 K_CS
    20.0,   # 12 K_EiS
    1.0,    # 13 K_EeS
    25.0,   # 14 K_act_S
    13.0,   # 15 K_inh_S
    1.0,    # 16 K_IS
    40.0,   # 17 K_DS
    120.0,  # 18 K_spike
    2000.0, # 19 K_spike_O
    1.0,    # 20 v_max_spike_D
    0.5,    # 21 v_max_spike_E
    0.5,    # 22 v_max_spike_C
    0.5,    # 23 v_max_spike_T
    0.3,    # 24 v_max_spike_O
    0.15,   # 25 k_transl_M
    0.75,   # 26 k_transl_S
    0.23,   # 27 k_deg_mRNA_M
    0.23,   # 28 k_deg_mRNA_S
    0.005,  # 29 k_deg_M
    0.001,  # 30 k_deg_E_int
    0.8,    # 31 k_deg_E_ext
    0.5,    # 32 k_deg_C
    0.0,    # 33 k_deg_T
    1e-7,   # 34 k_deg_S
    0.02,   # 35 k_deg_D
    0.02,   # 36 k_deg_O
    0.1,    # 37 k_echo_int
    1.0,    # 38 v_max_diff_D
    0.5,    # 39 v_max_diff_E
    0.4,    # 40 v_max_diff_C
    0.5,    # 41 v_max_diff_O
    100.0,  # 42 K_D_diff
    1.0,    # 43 K_OD
]

# 2. Network and diffusion configuration
n_bacteries = 27
d_max       = 3e-5
n_segments  = 3
R_cell      = 0.5e-6
dt          = 0.1

spacing = d_max / n_segments

gamma_D     = θ₀[34]   # k_deg_D
gamma_C     = θ₀[31]   # k_deg_C
gamma_E_ext = θ₀[30]   # k_deg_E_ext
gamma_O     = θ₀[35]   # k_deg_O
lambda      = 0.15 * spacing

species_names = [
    :D, :M, :I, :E_int, :T, :E_ext, :C, :O, :S,
    :D_diff, :E_ext_diff, :C_diff, :O_diff,
    :mRNA_M, :mRNA_S
]
n_species = length(species_names)

D_dict = Dict{Symbol, Float64}(
    :D => 0.0, :M => 0.0, :I => 0.0, :E_int => 0.0, :T => 0.0,
    :E_ext => 0.0, :C => 0.0, :O => 0.0, :S => 0.0,
    :D_diff     => lambda^2 * gamma_D,
    :E_ext_diff => lambda^2 * gamma_E_ext,
    :C_diff     => (0.4 * spacing)^2 * gamma_C,
    :O_diff     => (0.4 * spacing)^2 * gamma_O,
    :mRNA_M => 0.0, :mRNA_S => 0.0
)
gamma_dict = Dict{Symbol, Float64}(
    :D => 0.0, :M => 0.0, :I => 0.0, :E_int => 0.0, :T => 0.0,
    :E_ext => 0.0, :C => 0.0, :O => 0.0, :S => 0.0,
    :D_diff     => gamma_D,
    :E_ext_diff => gamma_E_ext,
    :C_diff     => gamma_C,
    :O_diff     => gamma_O,
    :mRNA_M => 0.0, :mRNA_S => 0.0
)
diffusion_targets = Dict{Symbol, Symbol}(
    :D => :D, :M => :M, :I => :I, :E_int => :E_int, :T => :T,
    :E_ext => :E_ext, :C => :C, :O => :O, :S => :S,
    :D_diff => :D, :E_ext_diff => :E_ext, :C_diff => :C, :O_diff => :O,
    :mRNA_M => :mRNA_M, :mRNA_S => :mRNA_S
)

# 3. Build the network
circuit, _ = BioKan.create_hebbian_non_spike_model(:node)

params_dict = Dict(parameters(circuit) .=> θ₀)
u0_raw      = Dict(s => 0.0 for s in species_names)
u0_dict     = BioKan.map_symbols_to_species(circuit, u0_raw)

net = BioKan.BioNetwork(d_max, n_bacteries)
BioKan.build_network_cube!(net, n_bacteries, d_max, n_segments,
                           circuit, params_dict, u0_dict, :ssa)
node_roles = BioKan.assign_conditioning_roles(net)

# S fixed at 40 for the Input/Output roles, random ∈ [20, 60] for interneurons
role_nodes = Set([node_roles[:Input_A], node_roles[:Input_B], node_roles[:Output]])
for id in keys(net.nodes)
    s_init = id in role_nodes ? 40.0 : Float64(round(Int, 20.0 + rand() * 40.0))
    BioKan.set_species!(net.nodes[id], :S, s_init)
end

@printf "Réseau   : %d bactéries  |  Input_A=%d  Input_B=%d  Output=%d\n" n_bacteries node_roles[:Input_A] node_roles[:Input_B] node_roles[:Output]
@printf "Espèces  : %d\n" n_species

weights = BioKan.compute_static_coupling_physics(
    net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

# 4. CS+/CS- protocol
max_density       = 300.0
stimulus_duration = 2000.0
interval          = 20000.0
delay_io          = 2000.0
force_duration    = 2000.0
eval_duration     = 5000.0

time_steps, Input_A, Input_B, Output_forced, mask, freeze_intervals =
    BioKan.pattern_to_learn_conditioning(;
        max_density       = max_density,
        stimulus_duration = stimulus_duration,
        interval          = interval,
        delay_io          = delay_io,
        force_duration    = force_duration,
        eval_duration     = eval_duration)
total_steps = length(time_steps)

@printf "Protocole : %d trials  |  steps=%d  |  T_total=%.0fs\n" length(mask) total_steps time_steps[end]
@printf "  stimulus=%.0fs  interval=%.0fs  delay_io=%.0fs  max_density=%.0f\n" stimulus_duration interval delay_io max_density
@printf "Phases de gel de S : %d fenêtres\n\n" length(freeze_intervals)

# 5. Spike threshold parameters
M_thr_spike  = 100.0
M_thr_O      = 200.0
t_refrac     = 5000.0
t_spike_emit = 2000.0
steps_refrac = round(Int, t_refrac  / dt)
steps_spike  = round(Int, t_spike_emit / dt)

amp_D_diff     = 300
amp_E_ext_diff = 300
amp_C_diff     = 30
amp_E_int      = 300
amp_O          = 30

println("Spike : M>$(M_thr_spike) → $(t_spike_emit)s d'émission ($(amp_D_diff)/step) | M>$(M_thr_O) → O | réfractaire=$(t_refrac)s\n")

# 6. Simulation loop
save_stride   = 1
n_saved       = ceil(Int, total_steps / save_stride)
history       = zeros(Float32, n_saved, n_bacteries, n_species)
saved_times   = zeros(Float64, n_saved)
save_idx      = 0

valid_nodes = [net.nodes[i] for i in 1:n_bacteries if haskey(net.nodes, i)]
valid_ids   = [i for i in 1:n_bacteries if haskey(net.nodes, i)]

ref_b        = valid_nodes[1]
all_sp_idx   = Int[get(ref_b.species_index, species_names[s], 0) for s in 1:n_species]
diff_s_list  = Int[s for s in 1:n_species if D_dict[species_names[s]] > 1e-40]
diff_src_idx = Int[get(ref_b.species_index, species_names[s], 0) for s in diff_s_list]
diff_tgt_idx = Int[get(ref_b.species_index, diffusion_targets[species_names[s]], 0) for s in diff_s_list]
diff_decay   = [exp(-gamma_dict[species_names[s]] * dt) for s in diff_s_list]
n_diff       = length(diff_s_list)

d_inj      = all_sp_idx[findfirst(==(:D),         species_names)]
m_sp_idx   = all_sp_idx[findfirst(==(:M),         species_names)]
o_sp_idx   = all_sp_idx[findfirst(==(:O),         species_names)]
d_diff_idx = all_sp_idx[findfirst(==(:D_diff),     species_names)]
e_diff_idx = all_sp_idx[findfirst(==(:E_ext_diff), species_names)]
c_diff_idx = all_sp_idx[findfirst(==(:C_diff),     species_names)]
o_diff_idx = all_sp_idx[findfirst(==(:O_diff),     species_names)]
e_int_sp_idx = all_sp_idx[findfirst(==(:E_int),    species_names)]

flux_emissions = zeros(Float64, n_bacteries, n_species)
received       = zeros(Float64, n_bacteries, n_species)

refrac_counter = zeros(Int, n_bacteries)
emit_counter   = zeros(Int, n_bacteries)
spike_counts   = zeros(Int, n_bacteries)

s_idx_sp    = all_sp_idx[findfirst(==(:S), species_names)]
forced_nodes = Set([node_roles[:Input_A], node_roles[:Input_B], node_roles[:Output]])
s_frozen     = zeros(Float64, n_bacteries)
in_freeze    = false
freeze_idx   = 1

b_in_A = net.nodes[node_roles[:Input_A]]
b_in_B = net.nodes[node_roles[:Input_B]]
b_out  = net.nodes[node_roles[:Output]]

CHECK_EVERY = 100
MAX_CONC    = 1e6

t_wall = time()
exploded, save_idx = let
    _exp     = false
    _save_idx = 0
    for step in 1:total_steps

        # A. External injection — Input_A, Input_B, Output teacher forcing
        @inbounds begin
            if Input_A[step] > 0.0
                b_in_A.integrator.u[d_inj] = round(Int, Input_A[step])
                BioKan.notify_bacterium!(b_in_A)
            end
            if Input_B[step] > 0.0
                b_in_B.integrator.u[d_inj] = round(Int, Input_B[step])
                BioKan.notify_bacterium!(b_in_B)
            end
            b_out.integrator.u[d_inj] = round(Int, Output_forced[step])
            BioKan.notify_bacterium!(b_out)
        end

        # B. Internal biology
        Threads.@threads for b in valid_nodes
            BioKan.step_bacterium!(b, dt)
        end

        # B'. Freeze S during the test windows
        t_now = time_steps[step]
        if freeze_idx <= length(freeze_intervals)
            fi = freeze_intervals[freeze_idx]
            if !in_freeze && t_now >= fi.t_start
                global in_freeze = true
                for (k, b) in enumerate(valid_nodes)
                    i = valid_ids[k]
                    s_frozen[i] = (i in forced_nodes) ? 40.0 : Float64(b.integrator.u[s_idx_sp])
                end
                @printf "Gel S #%d démarré à t=%.0fs\n" freeze_idx t_now
            end
            if in_freeze && t_now >= fi.t_end
                global in_freeze = false
                global freeze_idx += 1
                @printf "Gel S #%d terminé à t=%.0fs\n" (freeze_idx - 1) t_now
            end
        end
        if in_freeze
            for (k, b) in enumerate(valid_nodes)
                i = valid_ids[k]
                b.integrator.u[s_idx_sp] = round(Int, clamp(s_frozen[i], 0.0, 1e6))
            end
        end

        # C. Spike logic (sequential — avoids data races on shared state)
        for (k, b) in enumerate(valid_nodes)
            i     = valid_ids[k]
            M_val = b.integrator.u[m_sp_idx]
            O_val = b.integrator.u[o_sp_idx]

            if M_val >= M_thr_O
                b.integrator.u[o_diff_idx] = amp_O
                BioKan.notify_bacterium!(b)
            end

            if M_val >= M_thr_spike && O_val <= 5.0 &&
                    refrac_counter[i] == 0 && emit_counter[i] == 0
                emit_counter[i]   = steps_spike
                refrac_counter[i] = steps_spike + steps_refrac
                spike_counts[i]  += 1
            end

            if emit_counter[i] > 0
                if O_val >= 5.0
                    emit_counter[i] = 0
                    b.integrator.u[d_diff_idx]   = 0
                    b.integrator.u[e_diff_idx]   = 0
                    b.integrator.u[c_diff_idx]   = 0
                    b.integrator.u[e_int_sp_idx] = 0
                    BioKan.notify_bacterium!(b)
                else
                    b.integrator.u[d_diff_idx]   = amp_D_diff
                    b.integrator.u[e_diff_idx]   = amp_E_ext_diff
                    b.integrator.u[c_diff_idx]   = amp_C_diff
                    b.integrator.u[e_int_sp_idx] = amp_E_int
                    BioKan.notify_bacterium!(b)
                    emit_counter[i] -= 1
                end
            end

            refrac_counter[i] > 0 && (refrac_counter[i] -= 1)
        end

        # D. Collect diffusible fluxes
        fill!(flux_emissions, 0.0)
        @inbounds for k in eachindex(valid_nodes)
            i = valid_ids[k]
            u = valid_nodes[k].integrator.u
            for d in 1:n_diff
                v = Float64(u[diff_src_idx[d]])
                v > 0.0 && (flux_emissions[i, diff_s_list[d]] = v)
            end
        end

        # E. Transport (in-place)
        BioKan.propagate_signals_instantaneous!(
            received, weights, flux_emissions, n_bacteries, n_species)

        # F. Apply diffusion update
        for k in eachindex(valid_nodes)
            b = valid_nodes[k]
            i = valid_ids[k]
            u = b.integrator.u
            @inbounds for d in 1:n_diff
                amt = round(Int, max(0.0, received[i, diff_s_list[d]] * diff_decay[d]))
                amt > 0 && (u[diff_tgt_idx[d]] += amt)
                u[diff_src_idx[d]] = 0
            end
            BioKan.notify_bacterium!(b)
        end

        # G. Record history
        if step % save_stride == 0 && _save_idx < n_saved
            _save_idx += 1
            saved_times[_save_idx] = time_steps[step]
            for k in eachindex(valid_nodes)
                b = valid_nodes[k]
                i = valid_ids[k]
                u = b.integrator.u
                @inbounds for s in 1:n_species
                    history[_save_idx, i, s] = max(0.0f0, Float32(u[all_sp_idx[s]]))
                end
            end
        end

        # H. Explosion guard
        if step % CHECK_EVERY == 0 && _save_idx > 0 && maximum(@view history[_save_idx, :, :]) > MAX_CONC
            val, ci = findmax(@view history[_save_idx, :, :])
            @printf "[EXPLOSION] t=%.1f | bact=%d | espèce=%s | val=%.2e\n" time_steps[step] ci[1] species_names[ci[2]] val
            _exp = true
            break
        end
    end
    _exp, _save_idx
end
@printf "Simulation terminée en %.1f s\n" (time() - t_wall)

# 7. Loss and diagnostics
if !exploded
    d_idx = findfirst(==(:D), species_names)
    loss  = BioKan.compute_loss_conditioning(history, saved_times[1:save_idx], mask, node_roles, d_idx)
    @printf "\nLoss = %.4f  (réf aléatoire = %.4f)\n" loss log(2)
    println(loss < log(2) ? "Réponse sélective détectée." : "Pas de sélectivité (attendu avec θ par défaut).")
end

println("\n--- Spikes par bactérie (non nuls) ---")
for i in sort(collect(keys(net.nodes)))
    spike_counts[i] > 0 && @printf "  bact %2d : %d spikes\n" i spike_counts[i]
end

println("\n--- Moyenne S par bactérie ---")
s_hist_idx = findfirst(==(:S), species_names)
for id in sort(collect(keys(net.nodes)))
    @printf "  bact %2d : mean(S)=%.3f\n" id mean(history[:, id, s_hist_idx])
end

println("\n--- Max / Moyenne par espèce (réseau) ---")
for (s, name) in enumerate(species_names)
    mx = maximum(history[:, :, s])
    mx > 0.01 && @printf "  %-14s max=%8.2f  mean=%6.3f\n" name mx mean(history[:,:,s])
end

# 8. Plots
t           = saved_times[1:save_idx]
plot_stride = max(1, div(length(t), 5000))
tp          = t[1:plot_stride:end]

function downsample_max(trace, stride)
    stride == 1 && return trace
    return [maximum(view(trace, i:min(i+stride-1, length(trace)))) for i in 1:stride:length(trace)]
end

function add_eval_windows!(p)
    for trial in mask
        col = trial.expected == 1 ? RGBA(0.0,0.6,0.0,0.15) : RGBA(1.0,0.5,0.0,0.15)
        vspan!(p, [trial.t_start, trial.t_end], color=col, label="")
    end
    vline!(p, [-1.0], color=RGBA(0.0,0.6,0.0,0.6), lw=2, label="eval CS+")
    vline!(p, [-1.0], color=RGBA(1.0,0.5,0.0,0.6), lw=2, label="eval CS-")
end

d_plot    = findfirst(==(:D),     species_names)
m_plot    = findfirst(==(:M),     species_names)
ei_plot   = findfirst(==(:E_int), species_names)
ee_plot   = findfirst(==(:E_ext), species_names)
c_plot    = findfirst(==(:C),     species_names)
t_plot    = findfirst(==(:T),     species_names)
s_plot    = findfirst(==(:S),     species_names)
o_plot    = findfirst(==(:O),     species_names)
K_MD_plot = 140.0

t_full     = collect(time_steps)
inp_stride = save_stride * plot_stride
tp_full    = t_full[1:inp_stride:end]
inA_ds     = Input_A[1:inp_stride:length(tp_full)*inp_stride]
inB_ds     = Input_B[1:inp_stride:length(tp_full)*inp_stride]
out_ds     = Output_forced[1:inp_stride:length(tp_full)*inp_stride]
n_tp       = min(length(tp_full), length(inA_ds), length(inB_ds))

p_inputs = plot(tp_full[1:n_tp], inA_ds[1:n_tp], label="Input_A (CS+)", color=:blue, lw=1.5,
                title="Stimuli injectés", ylabel="mol", xlabel="Temps (s)",
                tickfontsize=10, guidefontsize=10)
plot!(p_inputs, tp_full[1:n_tp], inB_ds[1:n_tp], label="Input_B (CS-)", color=:orange, lw=1.5)
add_eval_windows!(p_inputs)

p_out_inj = plot(tp_full[1:n_tp], out_ds[1:n_tp], label="Output forcé", color=:teal, lw=1.5,
                 title="Output teacher forcing", ylabel="mol", xlabel="Temps (s)",
                 tickfontsize=10, guidefontsize=10)
add_eval_windows!(p_out_inj)

role_ids    = [node_roles[:Input_A], node_roles[:Input_B], node_roles[:Output]]
role_labels = ["Input_A CS+ ($(node_roles[:Input_A]))",
               "Input_B CS- ($(node_roles[:Input_B]))",
               "Output ($(node_roles[:Output]))"]
inter_pool  = [id for id in 1:n_bacteries if id ∉ role_ids]
inter_ids   = inter_pool[randperm(length(inter_pool))[1:min(3, length(inter_pool))]]
all_ids     = vcat(role_ids, inter_ids)
all_labels  = vcat(role_labels, ["Interneurone ($id)" for id in inter_ids])

panels_dmi  = [p_inputs, p_out_inj]
panels_zoom = [p_inputs, p_out_inj]

for (id, lab) in zip(all_ids, all_labels)
    d_trace  = downsample_max(Float64.(history[:, id, d_plot]),  plot_stride)
    m_trace  = downsample_max(Float64.(history[:, id, m_plot]),  plot_stride)
    ei_trace = downsample_max(Float64.(history[:, id, ei_plot]), plot_stride)
    ee_trace = downsample_max(Float64.(history[:, id, ee_plot]), plot_stride)
    c_trace  = downsample_max(Float64.(history[:, id, c_plot]),  plot_stride)
    t_trace  = downsample_max(Float64.(history[:, id, t_plot]),  plot_stride)
    s_trace  = downsample_max(Float64.(history[:, id, s_plot]),  plot_stride)
    o_trace  = downsample_max(Float64.(history[:, id, o_plot]),  plot_stride)

    @printf "\n[%s]\n" lab
    @printf "  D=%8.2f  M=%8.2f  E_int=%6.2f  E_ext=%6.2f\n" maximum(d_trace) maximum(m_trace) maximum(ei_trace) maximum(ee_trace)
    @printf "  C=%8.2f  T=%8.2f  S=%8.3f  O=%6.2f\n" maximum(c_trace) maximum(t_trace) maximum(s_trace) maximum(o_trace)

    p1 = plot(tp, d_trace, label="D", color=:steelblue, lw=1,
              title=lab, ylabel="mol", xlabel="Temps (s)",
              tickfontsize=10, guidefontsize=10)
    plot!(p1, tp, m_trace, label="M", color=:crimson, lw=1)
    hline!(p1, [M_thr_spike], color=:red,    lw=1, ls=:dash, label="M_thr=$(M_thr_spike)")
    hline!(p1, [M_thr_O],     color=:orange, lw=1, ls=:dash, label="M_thr_O=$(M_thr_O)")
    add_eval_windows!(p1)
    push!(panels_dmi, p1)

    clip = K_MD_plot
    d_cl  = min.(d_trace,  100.0)
    m_cl  = min.(m_trace,  clip)
    ei_cl = min.(ei_trace, clip)
    ee_cl = min.(ee_trace, clip)
    c_cl  = min.(c_trace,  clip)
    t_cl  = min.(t_trace,  clip)
    s_cl  = min.(s_trace,  clip)
    o_cl  = min.(o_trace,  clip)
    ymax  = max(maximum(d_cl), maximum(m_cl), maximum(ei_cl), maximum(ee_cl),
                maximum(c_cl), maximum(t_cl), maximum(s_cl), maximum(o_cl), 1.0)
    p2 = plot(tp, m_cl,  label="M",           color=:crimson,   lw=1,
              title="$(lab) — zoom", ylabel="mol", xlabel="Temps (s)",
              ylims=(0.0, ymax * 1.2), tickfontsize=10, guidefontsize=10)
    plot!(p2, tp, d_cl,  label="D (clip100)", color=:steelblue, lw=1)
    plot!(p2, tp, ei_cl, label="E_int",       color=:brown,     lw=1)
    plot!(p2, tp, ee_cl, label="E_ext",       color=:pink,      lw=1)
    plot!(p2, tp, c_cl,  label="C",           color=:purple,    lw=1)
    plot!(p2, tp, t_cl,  label="T",           color=:orange,    lw=1)
    plot!(p2, tp, s_cl,  label="S",           color=:black,     lw=1.5)
    plot!(p2, tp, o_cl,  label="O",           color=:teal,      lw=1)
    add_eval_windows!(p2)
    push!(panels_zoom, p2)
end

p_means = plot(title="Moyenne réseau par espèce", ylabel="mol", xlabel="Temps (s)",
               legend=:outertopright, tickfontsize=10, guidefontsize=10)
for (sp, col) in zip([:D, :M, :E_int, :E_ext, :C, :T, :S, :O],
                     [:steelblue, :crimson, :brown, :pink, :purple, :orange, :black, :teal])
    idx = findfirst(==(sp), species_names)
    isnothing(idx) && continue
    trace = downsample_max([Float64(mean(history[step,:,idx])) for step in axes(history,1)], plot_stride)
    plot!(p_means, tp, trace, label=string(sp), color=col, lw=1.5)
end
push!(panels_dmi, p_means, BioKan.plot_bionetwork_3d(net, node_roles))

ts_str  = Dates.format(now(), "yyyymmdd_HHMMSS")
n_dmi   = length(panels_dmi)
n_zoom  = length(panels_zoom)

fig_dmi  = plot(panels_dmi...,  layout=(n_dmi,  1), size=(1200, 300*n_dmi),  legend=:outertopright)
fig_zoom = plot(panels_zoom..., layout=(n_zoom, 1), size=(1200, 300*n_zoom), legend=:outertopright)

mkpath(joinpath(PROJECT_ROOT, "outputs"))
savefig(fig_dmi,  joinpath(PROJECT_ROOT, "outputs", "conditioning_$(ts_str)_DMI.png"))
savefig(fig_zoom, joinpath(PROJECT_ROOT, "outputs", "conditioning_$(ts_str)_zoom.png"))
println("\nDMI  : conditioning_$(ts_str)_DMI.png")
println("Zoom : conditioning_$(ts_str)_zoom.png")
