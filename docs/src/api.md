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

## Update/downdate

```@docs
PureSparse.SimplicialLDLFactor
PureSparse.simplicial
PureSparse.updowndate!
```
