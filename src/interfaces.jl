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

"""
    AbstractGame{S}

Supertype for games of size `S`.
"""
abstract type AbstractGame{S} end
const _AG = AbstractGame # alias

Base.size(::_AG{S}) where {S} = S

"""
    payoff(g::AbstractGame, x)

Compute the payoffs of `g` at the strategy profile `x`.
"""
function payoff(::_AG, ::Any) end


"""
    AbstractRegularizedGame{S, G <: AbstractGame} <: AbstractGame{S}

Supertype for regularized games of size `S` and type `G`.
"""
abstract type AbstractRegularizedGame{S, G <: _AG} <: _AG{S} end
const _ARG = AbstractRegularizedGame # alias


"""
    AbstractBaseGame{S, G <: AbstractGame} <: AbstractGame{S}

Supertype for base games of size `S` with corresponding latent game of type `G`.
"""
abstract type AbstractBaseGame{S, G <: _AG} <: _AG{S} end
const _ABG = AbstractBaseGame # alias


"""
    AbstractIterableScheme{G <: AbstractGame}

Supertype for iterable schemes for games of type `G`.
"""
abstract type AbstractIterableScheme{G <: _AG} end
const _AIS = AbstractIterableScheme # alias

function Base.iterate(::_AIS, ::Any) end
function Base.iterate(::_AIS) end

Base.IteratorSize(::Type{<: _AIS}) = Base.IsInfinite()