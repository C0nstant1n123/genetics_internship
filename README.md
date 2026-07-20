# BioKan

Julia framework for simulating **learning in bacterial communication networks**.
Each cell runs a stochastic gene-regulatory circuit (a Catalyst reaction
network integrated with the Gillespie SSA); cells couple through the physical
diffusion of signalling molecules; and an internal memory species implements a
Hebbian plasticity rule, letting a colony learn associations and logic gates.

## Structure

```
src/            Core model and simulation engine
  Bacterias.jl    Single-cell gene circuit + SSA/ODE stepping (Bacterium)
  Network.jl      Multicellular BioNetwork: geometry, edges, role assignment
  Diffusion.jl    Physics-based intercellular signal transport
  Hebbian.jl      Plasticity model, learning protocols, loss & truth tables
  BioKan.jl       Module entry point (re-exports the public API)
experiments/    The three experiments from the report (+ a 6-bacteria variant)
test/           Unit tests
```

## Experiments

Each script is self-contained: it builds the network, runs the training/test
protocol, prints a score, and saves diagnostic plots under `outputs/`.

| Script | What it does |
| --- | --- |
| `experiments/conditioning.jl` | CS+/CS- associative (Pavlovian) conditioning on a 27-bacteria cube. |
| `experiments/logic_gates.jl` | Learns a 2-input logic gate on an 8-bacteria Y topology. |
| `experiments/logic_gates_pacemaker.jl` | Same task with the reference input driven as a pacemaker. |
| `experiments/logic_gates_6bacteria.jl` | Simplified 6-bacteria variant (no interneurons). |

The logic-gate scripts take the target gate as their first argument (any key of
`BioKan.LOGIC_GATES`, e.g. `:XOR`, `:AND`, `:OR`, `:NAND`, …):

```
julia --project=. experiments/logic_gates.jl XOR
julia --project=. --threads=auto experiments/conditioning.jl
```

## Usage as a library

```julia
include("src/BioKan.jl")
using .BioKan
using Catalyst

circuit, defaults = create_hebbian_non_spike_model(:node)
params = Dict(parameters(circuit) .=> defaults)
u0     = map_symbols_to_species(circuit, Dict(:D => 0.0, :M => 0.0, :S => 0.0))

b = Bacterium(1, [0.0, 0.0], circuit, params, u0; mode=:ssa)
set_species!(b, :S, 50.0)
step_bacterium!(b, 1.0)
```

## Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Dependencies

Requires Julia ≥ 1.10. Install with:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Main dependencies: `Catalyst`, `JumpProcesses`, `OrdinaryDiffEq`,
`ModelingToolkit`, `Plots`.
