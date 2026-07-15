using Documenter, DocumenterCitations, DocumenterVitepress, PureSparse

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    sitename = "PureSparse.jl",
    authors = "el_oso",
    modules = [PureSparse],
    warnonly = true,
    plugins = [bib],
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
        "Sparse QR Guide" => "qr-guide.md",
        "Interior-Point Guide" => "ipm-guide.md",
        "Benchmarking" => "benchmarking.md",
        "API Reference" => "api.md",
        "Provenance & Licensing" => "provenance.md",
        "References" => "references.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureSparse.jl",
    devbranch = "master",
    push_preview = true,
)
