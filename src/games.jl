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

module Games

import Zygote: @adjoint, pullback
import StaticArrays: @MVector


export AbstractGame, MatchingPenniesGame, payoff


# DOCME
abstract type AbstractGame{S} end
const _AG = AbstractGame # alias

Base.size(::_AG{S}) where {S} = S

# DOCME
function payoff(::_AG, x) end


# DOCME 
struct MatchingPenniesGame <: _AG{(1, 1)} end
const _MPG = MatchingPenniesGame # alias

_u₁(::_MPG, (x₁, x₂)) = (2x₁ - 1) * (2x₂ - 1)

function payoff(g::_MPG, x)
  u₁_x = _u₁(g, x)
  @MVector [u₁_x, -u₁_x]
end

@adjoint function payoff(g::_MPG, x)
  payoff(g, x), function(∂x)
    _, ∂u₁ = pullback(_u₁, g, x)
    ∂x₁, ∂x₂ = ∂x
    ∂u₁_∂θ₁ =  ∂u₁(∂x₁)[2][1]
    ∂u₁_∂θ₂ =  ∂u₁(∂x₂)[2][2]
    ∂u = @MVector [∂u₁_∂θ₁, -∂u₁_∂θ₂]
    (nothing, ∂u)
  end
end


end # module Games