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

# DOCME
abstract type AbstractGame{S} end
const _AG = AbstractGame # alias

Base.size(::_AG{S}) where {S} = S

# DOCME
function payoff(::_AG, x) end


# DOCME
abstract type AbstractRegularizedGame{S, G <: _AG} <: _AG{S} end
const _ARG = AbstractRegularizedGame # alias


# DOCME
abstract type AbstractBaseGame{S, G <: _AG} <: _AG{S} end
const _ABG = AbstractBaseGame # alias


# DOCME
abstract type AbstractScheme{G <: _AG} end
const _AS = AbstractScheme # alias

# DOCME
function Base.iterate(::_AS, θₜ₋₁) end
function Base.iterate(::_AS) end

Base.IteratorSize(::Type{<: _AS}) = Base.IsInfinite()