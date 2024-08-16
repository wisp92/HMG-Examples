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

import StaticArrays: @MVector
import Zygote: @adjoint, pullback
import Reexport: @reexport

import ..Games: _AG
@reexport import ..Games: payoff


export AbstractRegularizedGame, L₂RegularizedGame


# DOCME
abstract type AbstractRegularizedGame{S, G <: _AG} <: _AG{S} end
const _ARG = AbstractRegularizedGame # alias

# DOCME
function game(::_ARG) end


# DOCME
struct L₂RegularizedGame{S, G} <: _ARG{S, G}
  g::G
  μ
  x₀
  function L₂RegularizedGame(g::G, μ=1e-4, x₀=zeros(sum(S))) where {S, G <: _AG{S}}
    new{S, G}(g, μ, x₀)
  end
end
const _L₂RG = L₂RegularizedGame # alias

game(g::_L₂RG) = g.g

function payoff(g::_L₂RG{S, <: Any}, x) where {S}
  L₂² = (x - g.x₀) .^ 2
  I = Iterators.Stateful(Base.OneTo(sum(S)))
  payoff(g.g, x) - g.μ * [sum(L₂²[Iterators.take(I, mᵢ) |> collect]) for mᵢ ∈ S]
end

@adjoint function payoff(g::_L₂RG{S, <: Any}, x) where {S}
  ∂L₂² = Iterators.Stateful(2(x - g.x₀))
  ∂R = g.μ .* [(Iterators.take(∂L₂², mᵢ) |> collect) for mᵢ ∈ S]
  payoff(g, x), function(∂x)
    _, ∂u = pullback(payoff, g.g, x)
    _, ∂u_∂x = ∂u(∂x)
    (nothing, ∂u_∂x - reduce(vcat, ∂x .* ∂R))
  end
end

end # module RegularizedGames

