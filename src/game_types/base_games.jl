# Copyright 2024 Iosif Sakos

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module BaseGames

import Zygote: @adjoint, pullback, hook

import .._AG, .._ABG, ..payoff


export BaseGame


@doc raw"""
    BaseGame{S, G} <: AbstractBaseGame{S, G}

Base game of size `S` with corresponding latent game of type `G`.

---

    BaseGame{S}(Γ::AbstractGame, χ) where {S}
  
Construct a `BaseGame{S, typeof(Γ)}` with corresponding latent game `Γ` and `χ` as the players' representation maps.

For each player ``i = 1, …, n``, the player's payoff ``uᵢ(θ)`` of the base game at the strategy profile ``θ ∈ Θ₁ × ⋯ × Θₙ`` is given by the formula:
```math
uᵢ(θ) = uᴴᵢ\big(χ₁(θ₁), …, χₙ(θₙ)\big),
```
where ``χⱼ : Θⱼ → 𝒳ⱼ`` for ``j = 1, …, n`` are the players' representation maps, and ``uᴴᵢ : 𝒳₁ × ⋯ × 𝒳ₙ → ℝ`` is the payoff function of player ``i`` in ``Γ``.
"""
struct BaseGame{S, G} <: _ABG{S, G}
  Γ::G
  χ
  function BaseGame{S}(Γ::G, χ) where {S, G}
    new{S, G}(Γ, χ)
  end
end
const _BG = BaseGame # alias

function payoff(g::_BG{S, <: Any}, θ) where {S}
  I = Iterators.Stateful(Base.OneTo(sum(S)))
  payoff(g.Γ, reduce(vcat, 
    χᵢ(θ[Iterators.take(I, mᵢ) |> collect]) 
    for (χᵢ, mᵢ) ∈ zip(g.χ, S)
  ))
end

@adjoint function payoff(g::_BG{S, <: Any}, θ; ∂=Dict()) where {S}
  I = Iterators.Stateful(Base.OneTo(sum(S)))
  I = [Iterators.take(I, mᵢ) |> collect for mᵢ ∈ S]
  χ = Tuple(begin
    ∂χᵢ = get(∂, χᵢ, nothing)
    isnothing(∂χᵢ) ? χᵢ : θᵢ -> χᵢ(hook(∂θᵢ -> (∂χᵢ[:] = ∂θᵢ), θᵢ))
  end for χᵢ ∈ g.χ)
  u = θ -> payoff(g.Γ, reduce(vcat, χᵢ(θ[Iᵢ]) for (χᵢ, Iᵢ) ∈ zip(χ, I)))
  payoff(g, θ), function(∂θ)
    _, ∂u = pullback(u, θ)
    (nothing, ∂u(∂θ) |> only)
  end
end


end # module BaseGames