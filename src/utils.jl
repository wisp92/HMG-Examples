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

module Utilities

import Random: AbstractRNG, default_rng
import StaticArrays: @MArray
import ChainRulesCore: @non_differentiable


export uniform, zeros


# DOCME
function uniform(rng::AbstractRNG, dims...; low=0, high=1, eltype=Float64)
  @assert low ≤ high
  scale = high - low
  W = @MArray rand(rng, eltype, dims...)
  W * scale .+ low
end

uniform(dims...; kwargs...) = 
  uniform(default_rng(), dims...; kwargs...)

uniform(rng::AbstractRNG=default_rng(); kwargs...) = 
  (dims...) -> uniform(rng, dims...; kwargs...)

@non_differentiable uniform(::Any...)


# DOCME
zeros(dims...; eltype=Float64) = @MArray zeros(eltype, dims...)
zeros(; kwargs...) = (dims...) -> zeros(dims...; kwargs...)

@non_differentiable zeros(::Any...)


end # module Utilities