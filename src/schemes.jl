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


module Schemes

import LinearAlgebra: pinv
import StaticArrays: SVector, MVector, sacollect
import Zygote: jacobian
import ChainRulesCore: @non_differentiable

import ..HiddenGames: _AHG, payoff
import ..Initializations: zeros as _zeros


export AbstractScheme, PreconditioningScheme


# DOCME
abstract type AbstractScheme{M, G <: _AHG} end
const _AS = AbstractScheme # alias

# DOCME
function Base.iterate(::_AS, θₜ₋₁) end
function Base.iterate(::_AS) end

Base.eltype(::Type{<: _AS{M}}) where {M} = MVector{M, Float64}
Base.IteratorSize(::Type{<: _AS}) = Base.IsInfinite()


# DOCME
struct PreconditioningScheme{M, G} <: _AS{M, G}
  g::G
  θ₀::MVector{M, Float64}
  γ::Float64
  function PreconditioningScheme(g::G, θ₀, γ=1e-4) where {S, G <: _AHG{S}}
    new{sum(S), G}(g, θ₀, γ)
  end
end
const _PS = PreconditioningScheme # alias


Base.iterate(s::_PS) = (s.θ₀, convert(SVector, s.θ₀))
function Base.iterate(s::_PS{M, <: Any}, θₜ₋₁) where {M}

  bg_sz = size(s.g)
  hg_sz = size(s.g.g) 
  n = length(bg_sz)

  ∂χ = sacollect(SVector{n}, _zeros(mᵢ, dᵢ) for (mᵢ, dᵢ) ∈ zip(bg_sz, hg_sz))
  f = Tuple(∂θᵢ -> (∂χᵢ[:] = ∂θᵢ) for ∂χᵢ ∈ ∂χ)
  payoff_f = θ -> payoff(s.g, θ; f=f)
  ∂u = jacobian(payoff_f, θₜ₋₁) |> only

  V = MVector{<: Any, Float64}[]
  sizehint!(V, n)
  jᵢ₋₁ = 0
  for (∂χᵢ, ∂uᵢ, mᵢ) ∈ zip(∂χ, eachrow(∂u), bg_sz)
    jᵢ = jᵢ₋₁ + mᵢ
    push!(V, pinv(∂χᵢ * ∂χᵢ') * ∂uᵢ[jᵢ₋₁ + 1:jᵢ])
    jᵢ₋₁ = jᵢ
  end

  θₜ = θₜ₋₁ + s.γ * reduce(vcat, V)
  convert(MVector, θₜ), θₜ
end

@non_differentiable iterate(::_PS, ::Any...)


end # module Schemes