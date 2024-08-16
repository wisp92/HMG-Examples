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

import LinearAlgebra: I as IdentityMatrix, Diagonal, pinv, svd
import StaticArrays: SVector, MVector, MMatrix
import Zygote: pullback, withjacobian

import ..Games: _AG, payoff
import ..BaseGames: _BG

import ..Initializations: zeros as _zeros


export AbstractScheme, PreconditioningScheme


# DOCME
abstract type AbstractScheme{G <: _AG} end
const _AS = AbstractScheme # alias

# DOCME
function Base.iterate(::_AS, θₜ₋₁) end
function Base.iterate(::_AS) end

Base.IteratorSize(::Type{<: _AS}) = Base.IsInfinite()


# DOCME
struct PreconditioningScheme{M, G} <: _AS{G}
  g::G
  θ₀::MVector{M, Float64}
  γ
  abstol
  function PreconditioningScheme(g::G, θ₀, γ=1e-4; abstol=1e-4) where {S, G <: _AG{S}}
    new{sum(S), G}(g, θ₀, γ, abstol)
  end
end
const _PS = PreconditioningScheme # alias

function _P(∂χᵢ; abstol=1e-4)
  F = svd(∂χᵢ)
  S = Iterators.takewhile(>(abstol), F.S) |> collect
  U = F.U[:, begin:length(S)]
  U * Diagonal(pinv.(S) .^ 2) * U'
end

Base.iterate(s::_PS) = (s.θ₀, convert(SVector, s.θ₀))

# function Base.iterate(s::_PS{M, <: _BG}, θₜ₋₁::SVector{M, Float64}) where {M}
#   bg_sz = size(s.g)
#   hg_sz = size(s.g.g) 
#   n = length(bg_sz)

#   ∂χₜ₋₁ = Tuple(_zeros(mᵢ, dᵢ) for (mᵢ, dᵢ) ∈ zip(bg_sz, hg_sz))
#   IM = MMatrix{n, n, Int}(IdentityMatrix)
#   I = Iterators.Stateful(Base.OneTo(M))
#   ∂buₜ₋₁ =  Tuple(begin
#     _, ∂uᵢ = pullback(θ -> payoff(s.g, θ; ∂=Dict(χᵢ => ∂χᵢ)), θₜ₋₁)
#     (∂uᵢ(IMᵢ) |> only)[Iterators.take(I, mᵢ) |> collect]
#   end for (χᵢ, ∂χᵢ, mᵢ, IMᵢ) ∈ zip(s.g.χ, ∂χₜ₋₁, bg_sz, eachcol(IM)))

#   ∂uₜ₋₁ = reduce(vcat, 
#     _P(∂χᵢ; abstol=s.abstol) * ∂buᵢ
#     for (∂χᵢ, ∂buᵢ) ∈ zip(∂χₜ₋₁, ∂buₜ₋₁)
#   )

#   θₜ = θₜ₋₁ + s.γ * ∂uₜ₋₁
#   convert(MVector, θₜ), θₜ
# end

function Base.iterate(s::_PS{M, <: _BG}, θₜ₋₁::SVector{M, Float64}) where {M}
  bg_sz = size(s.g)
  hg_sz = size(s.g.g) 
  n = length(bg_sz)

  Θ = Iterators.Stateful(θₜ₋₁)
  O = Tuple(
    withjacobian(χᵢ, Iterators.take(Θ, mᵢ) |> collect) 
    for (χᵢ, mᵢ) ∈ zip(s.g.χ, bg_sz)
  )
  ∂χₜ₋₁ = Tuple(Oᵢ.grad |> only for Oᵢ ∈ O)

  xₜ₋₁ = MVector{sum(hg_sz), Float64}(reduce(vcat, Oᵢ.val for Oᵢ ∈ O))
  _, ∂hu = pullback(payoff, s.g.g, xₜ₋₁)
  IM = MMatrix{n, n, Int}(IdentityMatrix)
  I = Iterators.Stateful(Base.OneTo(sum(hg_sz)))
  ∂uₜ₋₁ = reduce(vcat, begin 
    (_, ∂huᵢ) = ∂hu(IMᵢ) 
    pinv(∂χᵢ; atol=s.abstol) * ∂huᵢ[Iterators.take(I, dᵢ) |> collect]
  end for (∂χᵢ, dᵢ, IMᵢ) ∈ zip(∂χₜ₋₁, hg_sz, eachcol(IM)))

  θₜ = θₜ₋₁ + s.γ * ∂uₜ₋₁
  convert(MVector, θₜ), θₜ
end

Base.eltype(::_PS{M, <: Any}) where {M} = MVector{M, Float64}


end # module Schemes