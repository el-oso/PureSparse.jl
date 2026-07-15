# API Reference

## Symbolic analysis

```@docs
PureSparse.symbolic
PureSparse.Symbolic
```

## Orderings

```@docs
PureSparse.AbstractOrdering
PureSparse.order
PureSparse.order_columns
PureSparse.AMDOrdering
PureSparse.NaturalOrdering
PureSparse.GivenOrdering
```

## Numeric factorization

```@docs
PureSparse.AbstractSparseFactor
PureSparse.cholesky
PureSparse.cholesky!
PureSparse.ldlt
PureSparse.ldlt!
PureSparse.SupernodalFactor
PureSparse.LDLFactor
PureSparse.FactorStats
PureSparse.issuccess
```

## Solving

```@docs
PureSparse.solve!
PureSparse.solve_L!
PureSparse.solve_D!
PureSparse.solve_Lt!
PureSparse.refine!
```

## Sparse QR

```@docs
PureSparse.symbolic_qr
PureSparse.qr
PureSparse.qr!
PureSparse.qr_frontal
PureSparse.apply_Q!
PureSparse.apply_Qt!
PureSparse.solve_R!
PureSparse.solve_Rt!
PureSparse.solve_minnorm!
PureSparse.QRSymbolic
PureSparse.QRFactor
PureSparse.QRStats
PureSparse.COLAMDOrdering
```

### [QR tuning constants](@id QR_AUTO_METHOD_RATIO)

`QR_AUTO_METHOD_RATIO` (default `40.0`, Preferences key `qr_auto_method_ratio` in
`src/tuning.jl`) is the `sym.flops / sym.nnzR` threshold above which
`qr(A; method = :auto)` selects `:frontal` over `:column` — see
[`qr`](@ref PureSparse.qr) and the [Sparse QR Guide](qr-guide.md) for how it was
calibrated.

### QR internals (referenced by the docstrings above)

Not part of the public API — included so the cross-references above resolve.

```@docs
PureSparse._qr_threshold
PureSparse._qr_block
```

## Update/downdate

```@docs
PureSparse.SimplicialLDLFactor
PureSparse.simplicial
PureSparse.updowndate!
```
