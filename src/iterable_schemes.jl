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

module IterableSchemes

import LinearAlgebra: I as IdentityMatrix, pinv
import StaticArrays: SVector, MVector, MMatrix
import Zygote: pullback, withjacobian

import .._AG, .._ABG, ..payoff, .._AIS
import ..BaseGames: _BG

import ..Utilities: zeros as _zeros


export LatentGradientAscent


@doc raw"""
    LatentGradientAscent{M, G <: AbstractBaseGame} <: AbstractIterableScheme{G}

Gradient ascent scheme for a base game of type `G` and input length `M`.

---

    LatentGradientAscent(Γ::AbstractBaseGame, θ₀[, γ]; <keyword arguments>)

Construct a `LatentGradientAscent{sum(size(Γ)), typeof(Γ)}` with step size `γ` initialized at the strategy profile `θ₀`.

The scheme is designed to provide optimal last-iterate convergence to the set of Nash equilibria of ``Γ`` under the assumption the corresponding latent game is strongly monotone.

# Extended Help

At each iteration ``t``, the scheme features a gradient step from the current strategy profile ``θₜ ∈ Θ₁ × ⋯ × Θₙ`` to a new strategy profile ``θₜ₊₁`` given, for each player ``i = 1, …, n``, by the formula:
```math
  [θₜ₊₁]ᵢ = [θₜ]ᵢ + γ [𝐏ₜ]ᵢ  ∇_{θᵢ} uᵢ(θ) \big|_{θ = θₜ},
```
where ``uᵢ(θ)`` is the player's payoff at the strategy profile ``θ`` in ``Γ``, and ``[𝐏ₜ]ᵢ`` is a preconditioning matrix given by the formula: 
```math
    [𝐏ₜ]ᵢ = \big(𝐉ᵢ([θₜ]ᵢ) 𝐉ᵢ([θₜ]ᵢ)ᵀ\big)⁺, 
```
where ``𝐉ᵢ`` is the Jacobian matrix of the player's representation map ``χᵢ : Θᵢ → 𝒳ᵢ``.

If the payoff function ``uᴴᵢ: 𝒳₁ × ⋯ × 𝒳ₙ → ℝ`` of ``i`` in the latent game is known, then ``[θₜ₊₁]ᵢ`` is equivalent to:
```math
  [θₜ₊₁]ᵢ = [θₜ]ᵢ + γ 𝐉ᵢ([θₜ]ᵢ)⁺ ∇_{xᵢ} uᴴᵢ(x) \big|_{x₁ = χ₁([θₜ]₁), …, xₙ = χₙ([θₜ]ₙ)}.
```
The latter formula features improved computational stability over the former one.

# Arguments
- `atol = 0`: Absolute tolerance assumed for the computation of the Moore-Penrose inverse.
"""
struct LatentGradientAscent{M, G <: _ABG} <: _AIS{G}
  Γ::G
  θ₀::MVector{M, Float64}
  γ
  atol
  function LatentGradientAscent(Γ::G, θ₀, γ=1e-4; 
    atol=0
  ) where {S, G <: _AG{S}}
    new{sum(S), G}(Γ, θ₀, γ, atol)
  end
end
const _LGA = LatentGradientAscent # alias

Base.iterate(s::_LGA) = (s.θ₀, convert(SVector, s.θ₀))

function Base.iterate(s::_LGA{<: Any, <: _BG}, θₜ₋₁)
  bg_sz = size(s.Γ)
  hg_sz = size(s.Γ.Γ) 
  n = length(bg_sz)

  Θ = Iterators.Stateful(θₜ₋₁)
  O = Tuple(
    withjacobian(χᵢ, Iterators.take(Θ, mᵢ) |> collect) 
    for (χᵢ, mᵢ) ∈ zip(s.Γ.χ, bg_sz)
  )
  ∂χₜ₋₁ = Tuple(Oᵢ.grad |> only for Oᵢ ∈ O)

  xₜ₋₁ = MVector{sum(hg_sz), Float64}(reduce(vcat, Oᵢ.val for Oᵢ ∈ O))
  _, ∂hu = pullback(payoff, s.Γ.Γ, xₜ₋₁)
  IM = MMatrix{n, n, Int}(IdentityMatrix)
  I = Iterators.Stateful(Base.OneTo(sum(hg_sz)))
  ∂uₜ₋₁ = reduce(vcat, begin 
    (_, ∂huᵢ) = ∂hu(IMᵢ) 
    pinv(∂χᵢ; atol=s.atol) * ∂huᵢ[Iterators.take(I, dᵢ) |> collect]
  end for (∂χᵢ, dᵢ, IMᵢ) ∈ zip(∂χₜ₋₁, hg_sz, eachcol(IM)))

  θₜ = θₜ₋₁ + s.γ * ∂uₜ₋₁
  convert(MVector, θₜ), convert(SVector, θₜ)
end

Base.eltype(::_LGA{M, <: Any}) where {M} = MVector{M, Float64}


end # module IterableSchemes