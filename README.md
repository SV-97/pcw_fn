Generic piecewise function trait and impls
===========================

pcw_fn is a library for handling piecewise defined functions (or other "piecewise
data") generically.

Currently the `PcwFn` trait the library is built around is implemented for
a single type: `VecPcwFn`. This type represents a piecewise function f given
by a collection of jump positions and functions such that
```
       ╭ f₁(x)   if      x < x₀
       │ f₂(x)   if x₀ ≤ x < x₁
f(x) = ┤ f₃(x)   if x₁ ≤ x < x₂
       │  ⋮               ⋮
       ╰ fₙ(x)   if xₙ ≤ x
```

for all x ∈ X where
    f₁,...,fₙ : X -> Y, and
    x₀ < x₁ < ... < xₙ
from some strictly totally ordered set X (so X is `Ord`). Note that for most functionality it's not required that the funcs actually behave like a function so the type can be used very well for "non-functional" `funcs`.

It's trivial to add impls backed by other types like `SmallVec` or `StaticVec`.

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>
