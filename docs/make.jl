using Documenter, DocumenterVitepress, PureSparse

# `remotes = nothing`: this repo has no `origin` configured yet (not pushed to GitHub),
# so Documenter can't auto-detect source-permalink URLs from `git remote`. Remove once the
# repo is actually pushed — `repo =` below (the VitePress deploy target, a separate
# setting from Documenter's own source-permalink `repo`) already names the intended URL.
makedocs(;
    sitename = "PureSparse.jl",
    authors = "el_oso",
    modules = [PureSparse],
    warnonly = true,
    remotes = nothing,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/el-oso/PureSparse.jl",
        devbranch = "master",
        devurl = "dev",
    ),
    draft = false,
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Benchmarking" => "benchmarking.md",
        "API Reference" => "api.md",
        "Provenance & Licensing" => "provenance.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureSparse.jl",
    devbranch = "master",
    push_preview = true,
)
