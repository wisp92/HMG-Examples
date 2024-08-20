using Documenter
using HMGExamples

makedocs(
  sitename="HMGExamples.jl",
  pages = [
    "Home" => "index.md",
    "Manual" => [
      "Games" => "games.md",
      "Iterable Schemes" => "iterable_schemes.md"
    ],
    "API" => "API.md"
  ]
)