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

import .._AG, ..payoff


export MatchingPenniesGame


@doc raw"""
    MatchingPenniesGame <: AbstractGame{(1, 1)}

Game of Matching Pennies.

The implementation differs from the typical one in that the game is realized in the strategy space ``X ≡ [0, 1]²``.

Specifically, for each ``x = (x₁, x₂) ∈ X``, ``xᵢ`` is the probability with which player ``i`` plays their first strategy, and therefore the payoffs ``u₁(x)`` and ``u₂(x)`` of the players at ``x`` are given by the formulas:

```math
  u₁(x) = -u₂(x) = (2x₁ - 1) * (2x₂ - 1).
```

---

    MatchingPenniesGame()

Construct a game of Matching Pennies.
"""
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