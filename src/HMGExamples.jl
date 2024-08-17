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

module HMGExamples

export Utilities
export AbstractGame, AbstractRegularizedGame, AbstractBaseGame, AbstractScheme, payoff
export Games, RegularizedGames, BaseGames, Schemes
export L₂, hide


include("utils.jl")
include("interfaces.jl")

include("game_types/games.jl")
include("game_types/regularized_games.jl")
include("game_types/base_games.jl")

include("schemes.jl")


import .RegularizedGames: L₂RegularizedGame
import .BaseGames: BaseGame

# DOCME
L₂(g::_AG{S}, μ=1e-4; x₀=zeros(sum(S))) where {S} = L₂RegularizedGame(g, μ, x₀)

# DOCME
hide(g::_AG, χ, dims::Tuple{Vararg{Int}}) = BaseGame{dims}(g, χ)


end # module HMGExamples
