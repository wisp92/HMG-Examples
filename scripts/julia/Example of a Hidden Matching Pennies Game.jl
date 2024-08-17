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
using HMGExamples.Utilities
using HMGExamples.Games
using HMGExamples.Schemes

const zeros = HMGExamples.Utilities.zeros


function representation_map(dims; init=uniform, bias=zeros, activator=celu)
  layers = Dense[]
  sizehint!(layers, 1 + length(dims))
  prev_dim, state = iterate(dims) # We assume `length(dims) > 0`.
  while true
    next = iterate(dims, state)
    if isnothing(next) break end
    dim, state = next
    push!(layers, Dense(prev_dim => dim, activator; init=init, bias=bias(dim)))
    prev_dim = dim
  end
  push!(layers, Dense(prev_dim => 1, sigmoid; init=init, bias=bias(1)))
  Chain(layers)
end


Random.seed!(0)

(m₁, m₂) = S = (1, 1)
χ₁ = representation_map([m₁, 1]; 
  init=uniform(; low=-1, high=1), 
  activator=x -> celu(x, 10)
)
χ₂ = representation_map([m₂, 1]; 
  init=uniform(; low=-1, high=1),
  activator=x -> celu(x, 10)
)

hg = L₂(MatchingPenniesGame(), 4e-2; x₀=fill(0.5, 2))
bg = hide(hg, (χ₁, χ₂), S)
# θ₀ = uniform(m₁ + m₂; low=-1, high=1)
θ₀ = @MVector [1.25, 2.25]
s = PreconditioningScheme(bg, θ₀, 1e-2; abstol=0)
θ = Iterators.take(s, 40000) |> collect
x = [[χ₁(θₜ[begin:m₁]); χ₂(θₜ[m₁ + 1:m₁ + m₂])] for θₜ ∈ θ]

