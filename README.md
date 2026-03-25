# BioKan

Julia framework for simulating bacterial communication networks with stochastic reaction kinetics.

## Structure

```
src/          Core modules (Bacterias, Network, Diffusion)
tests/        Test scripts for each model
scripts/      Batch simulation, config generation, rate-distortion analysis
```

## Models

- **Burst circuit** — oscillatory gene network with Hill kinetics (X/Y/Z + mRNA)
- **Hebbian model** — rate-coded plasticity model (D/M/I/C/T/E_ext/E_int)
- **iFFL** — incoherent feed-forward loop
- **Hill repeaters** — input/output relay nodes

## Usage

```julia
include("src/BioKan.jl")
using .BioKan

circuit, p_defaults = create_burst_circuit(:my_node)
b = Bacterium(1, [0.0, 0.0], circuit, Dict(parameters(circuit) .=> p_defaults), u0; mode=:ssa)
step_bacterium!(b, dt)
```

See `tests/test_burst.jl` and `tests/test_hebbian.jl` for full examples.

## Dependencies

Requires Julia ≥ 1.10. Install with:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Main dependencies: `Catalyst`, `JumpProcesses`, `OrdinaryDiffEq`, `ModelingToolkit`, `Plots`.
