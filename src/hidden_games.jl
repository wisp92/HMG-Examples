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


module HiddenGames

import StaticArrays: @MVector
import Zygote: @adjoint, pullback, hook
import Reexport: @reexport

import ..Games: _AG, _MPG
@reexport import ..Games: payoff, _u₁ as _u₁_x


export AbstractHiddenGame, HiddenMatchingPenniesGame


# DOCME
abstract type AbstractHiddenGame{S, G <: _AG} <: _AG{S} end
const _AHG = AbstractHiddenGame # alias


# DOCME
struct HiddenMatchingPenniesGame{S} <: _AHG{S, _MPG}
  g::_MPG
  χ₁
  χ₂
  function HiddenMatchingPenniesGame{S}(χ₁, χ₂) where {S}
    @assert S isa Tuple{Int, Int}
    new{S}(_MPG(), χ₁, χ₂)
  end
end
const _HMPG = HiddenMatchingPenniesGame # alias

@generated function payoff(g::_HMPG{S}, θ) where {S}
  m₁, m₂ = S
  I₁ = 1:m₁
  I₂ = (m₁ + 1):(m₁ + m₂)
  quote
    payoff(g.g, (g.χ₁(θ[$I₁]) |> only, g.χ₂(θ[$I₂]) |> only))
  end
end

@adjoint function payoff(g::_HMPG{S}, θ; f=(identity, identity)) where {S}
  χ₁, χ₂ = Tuple(θᵢ -> χᵢ(hook(fᵢ, θᵢ)) for (χᵢ, fᵢ) ∈ zip((g.χ₁, g.χ₂), f))
  _u₁_θ = θ -> _u₁_x(g.g, [χ₁(θ[I₁]) |> only; χ₂(θ[I₂]) |> only])
  m₁, m₂ = S
  I₁ = 1:m₁
  I₂ = (m₁ + 1):(m₁ + m₂)
  payoff(g, θ), function(∂θ)
    _, ∂u₁ = pullback(_u₁_θ, θ)
    ∂u₁₁_∂θ, ∂u₁₂_∂θ = ∂u₁.(∂θ) .|> only
    ∂u = [∂u₁₁_∂θ[I₁]; -∂u₁₂_∂θ[I₂]]
    (nothing, ∂u)
  end
end


end # module HiddenGames