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


using Random
using StaticArrays
using Flux

using HMGExamples
using HMGExamples.Initializations
using HMGExamples.HiddenGames
using HMGExamples.Schemes

const zeros = HMGExamples.Initializations.zeros


function representation_map(dims; init=uniform, bias=zeros)
  layers = Dense[]
  sizehint!(layers, 1 + length(dims))
  prev_dim, state = iterate(dims) # We assume `length(dims) > 0`.
  while true
    next = iterate(dims, state)
    if isnothing(next) break end
    dim, state = next
    push!(layers, Dense(prev_dim => dim, celu; init=init, bias=bias(dim)))
    prev_dim = dim
  end
  push!(layers, Dense(prev_dim => 1, sigmoid; init=init, bias=bias(1)))
  Chain(layers)
end


Random.seed!(0)

(m₁, m₂) = S = (2, 2)
χ₁ = representation_map([m₁, 2]; init=uniform(; low=0, high=1))
χ₂ = representation_map([m₂, 2]; init=uniform(; low=0, high=1))

g = HiddenMatchingPenniesGame{S}(χ₁, χ₂)
θ₀ = @MVector [-2.0, 2.0, -3.0, 3.0]
s = PreconditioningScheme(g, θ₀)
collect(Iterators.take(s, 100))
