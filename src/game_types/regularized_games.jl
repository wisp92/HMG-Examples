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

module RegularizedGames

import Zygote: @adjoint, pullback

import .._AG, .._ARG, ..payoff


export L₂RegularizedGame


@doc raw"""
    L₂RegularizedGame{S, G} <: AbstractRegularizedGame{S, G}

Game of size `S` and type `G` regularized based on the L₂-norm.

---

    L₂RegularizedGame(Γ::AbstractGame, μ[, x₀])

Construct a `L₂RegularizedGame{size(Γ), typeof(Γ)}` with weight `μ` and reference point `x₀`.

For each player ``i``, the players' payoff ``ũᵢ(x)`` of the regularized game at the strategy profile ``x`` is given by the formula:
```math
  ũᵢ(x) = uᵢ(x) - μ ∑_{j ∈ Sᵢ} (xⱼ - [x₀]ⱼ)²,
```
where ``uᵢ(x)`` is the payoff of ``i`` at ``x`` in `Γ`, and ``Sᵢ`` are the indices of ``x`` corresponding to the strategies of ``i``.
The argument `μ` and the optional argument `x₀` correspond respectively to the weight of the regularizer and a reference point from which the distance to ``x`` is measured.
If `x₀` is not specified, the origin `x₀ = 0` is used as a reference point.
"""
struct L₂RegularizedGame{S, G} <: _ARG{S, G}
  Γ::G
  μ
  x₀
  function L₂RegularizedGame(Γ::G, μ, x₀=zeros(sum(S))) where {S, G <: _AG{S}}
    new{S, G}(Γ, μ, x₀)
  end
end
const _L₂RG = L₂RegularizedGame # alias

function payoff(g::_L₂RG{S, <: Any}, x) where {S}
  L₂² = (x - g.x₀) .^ 2
  I = Iterators.Stateful(Base.OneTo(sum(S)))
  payoff(g.Γ, x) - g.μ * [sum(L₂²[Iterators.take(I, mᵢ) |> collect]) for mᵢ ∈ S]
end

@adjoint function payoff(g::_L₂RG{S, <: Any}, x) where {S}
  ∂L₂² = Iterators.Stateful(2(x - g.x₀))
  ∂R = g.μ .* [(Iterators.take(∂L₂², mᵢ) |> collect) for mᵢ ∈ S]
  payoff(g, x), function(∂x)
    _, ∂u = pullback(payoff, g.Γ, x)
    _, ∂u_∂x = ∂u(∂x)
    (nothing, ∂u_∂x - reduce(vcat, ∂x .* ∂R))
  end
end


end # module RegularizedGames

