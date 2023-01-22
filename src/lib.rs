//! Generic piecewise functions that allow for different internal buffers.
//!
//! # Examples
//! ```
//! use pcw_fn::{VecPcwFn, PcwFn};
//!
//! let pcw_poly: VecPcwFn<_, &dyn Fn(i32) -> i32> = VecPcwFn::try_from_iters(
//!     vec![5, 10, 15],
//!     vec![
//!         &(|x| x-10) as &dyn Fn(i32) -> i32,
//!         &(|x| 10) as &dyn Fn(i32) -> i32,
//!         &(|x| x*x + 5) as &dyn Fn(i32) -> i32,
//!         &(|x| x) as &dyn Fn(i32) -> i32,
//!     ],
//! )
//! .unwrap();
//! assert_eq!(pcw_poly.eval(0), -10);
//! assert_eq!(pcw_poly.eval(12), 12*12 + 5);
//! assert_eq!(pcw_poly.eval(500), 500);
//!
//! let pcw_const = VecPcwFn::try_from_iters(
//!     vec![5, 10, 15],
//!     vec!["hi", "how", "are", "you"],
//! )
//! .unwrap();
//! assert_eq!(pcw_const.func_at(&0), &"hi");
//!
//! let f: VecPcwFn<_, String> = pcw_const.combine(VecPcwFn::global(2), |s, x| format!("{s} {x}"));
//! assert_eq!(f.func_at(&15), &"you 2");
//! ```
//!
//! TODO: Add SmallVec and StaticVec backed variants.
//! TODO: Add normalization / removal of redundant jumps.
//! TODO: remove `is_sorted` dependency once `is_sorted` is stabilized.

// #![feature(generic_const_exprs)]
use std::cmp::Ordering;
use std::iter;

use itertools::{EitherOrBoth, Itertools};

mod functional_hackery;
pub use crate::functional_hackery::{Functor, FunctorRef, Kind1To1};
use is_sorted::IsSorted;

/// The different errors that can occur when constructing a piecewise function:
/// * The jumps aren't strictly sorted (in particular there might be duplicates).
/// * There's too many jumps; a piecewise function always has to have exactly one
/// jump less than it has functions.
/// * There's not enough jumps for the given number of functions.
#[derive(Debug)]
pub enum PcwFnError {
    JumpsNotStrictlySorted,
    TooManyJumpsForFuncs,
    TooFewJumpsForFuncs,
}

/// A piecewise function given by
///        ╭ f₁(x)   if      x < x₀
///        │ f₂(x)   if x₀ ≤ x < x₁
/// f(x) = ┤ f₃(x)   if x₁ ≤ x < x₂
///        │  ⋮               ⋮
///        ╰ fₙ(x)   if xₙ ≤ x
/// for all x ∈ X where
///     f₁,...,fₙ : X -> Y, and
///     x₀ < x₁ < ... < xₙ
/// from some strictly totally ordered set X (so X is `Ord`). Note that the fᵢ are not
/// necessarily distinct.
///
/// We'll call the collection of all xᵢ the jump positions, or simply jumps of the piecewise
/// function.
pub trait PcwFn<X: Ord, F>: Functor<F> + Sized {
    type JmpIter: Iterator<Item = X>;
    type FncIter: Iterator<Item = F>;

    /// Get a reference to the jumps.
    fn jumps(&self) -> &[X];

    /// Get a reference to the funcs in order.
    fn funcs(&self) -> &[F];

    /// Get a mutable reference to the jumps.
    fn funcs_mut(&mut self) -> &mut [F];

    /// Try constructing a new piecewise function from iterators over jumps and functions.
    fn try_from_iters<Jmp: IntoIterator<Item = X>, Fnc: IntoIterator<Item = F>>(
        jumps: Jmp,
        funcs: Fnc,
    ) -> Result<Self, PcwFnError>;

    /// A function that's globally given by a single function `f`.
    fn global(f: F) -> Self {
        Self::try_from_iters(iter::empty(), iter::once(f)).unwrap()
    }

    /// Add another segment to the piecewise function at the back.
    fn add_segment(&mut self, jump: X, func: F);

    /// Deconstruct a piecewise function into sequences of functions and jumps.
    fn into_jumps_and_funcs(self) -> (Self::JmpIter, Self::FncIter);

    /// Turn the function into an owned iterator over the jumps.
    fn into_jumps(self) -> Self::JmpIter {
        self.into_jumps_and_funcs().0
    }

    /// Turn the function into an owned iterator over the functions.
    fn into_funcs(self) -> Self::FncIter {
        self.into_jumps_and_funcs().1
    }

    /// How many segments the function consists of.
    fn segment_count(&self) -> usize {
        self.funcs().len()
    }

    /// Combine two piecewise functions using a pointwise action to obtain another piecewise function.
    fn combine<Rhs, G, Out, H>(self, rhs: Rhs, mut action: impl FnMut(F, G) -> H) -> Out
    where
        X: Ord,
        F: Clone,
        G: Clone,
        Rhs: PcwFn<X, G>,
        Out: PcwFn<X, H>,
    {
        match (self.segment_count(), rhs.segment_count()) {
            (0, _) => panic!("Empty function is invalid"),
            (_, 0) => panic!("Empty function is invalid"),
            (1, _) => {
                let l = self.into_funcs().next().unwrap();
                let (jr, fr) = rhs.into_jumps_and_funcs();
                Out::try_from_iters(jr, fr.map(|r| action(l.clone(), r))).unwrap()
            }
            (_, 1) => {
                let r = rhs.into_funcs().next().unwrap();
                let (jl, fl) = self.into_jumps_and_funcs();
                Out::try_from_iters(jl, fl.map(|l| action(l, r.clone()))).unwrap()
            }
            (n, m) => {
                let (jl, mut fl) = self.into_jumps_and_funcs();
                let (jr, mut fr) = rhs.into_jumps_and_funcs();
                // there'll be no more than n+m segments in the combined function
                let mut funcs = Vec::with_capacity(n + m);
                // -1 is always valid since we know fl and fr have at least one element
                let mut jumps = Vec::with_capacity(funcs.capacity() - 1);
                let mut l = fl.next().unwrap();
                let mut r = fr.next().unwrap();
                funcs.push(action(l.clone(), r.clone())); // value of result sufficiently far left in the domain
                for c in jl.merge_join_by(jr.into_iter(), std::cmp::Ord::cmp) {
                    match c {
                        EitherOrBoth::Left(jump) => {
                            jumps.push(jump);
                            l = fl.next().unwrap_or(l);
                        }
                        EitherOrBoth::Right(jump) => {
                            jumps.push(jump);
                            r = fr.next().unwrap_or(r);
                        }
                        EitherOrBoth::Both(jump, _) => {
                            jumps.push(jump);
                            l = fl.next().unwrap_or(l);
                            r = fr.next().unwrap_or(r);
                        }
                    }
                    funcs.push(action(l.clone(), r.clone()));
                }
                jumps.shrink_to_fit();
                funcs.shrink_to_fit();
                Out::try_from_iters(jumps.into_iter(), funcs.into_iter()).unwrap()
            }
        }
    }

    /// Resample `self` to the segments of `other`: replace the jumps of `self` with those of
    /// other and if that leaves multiple functions on a single segment combine them using the
    /// provided `combine` function.
    fn resample_to<PcwOut, G>(
        self,
        other: impl PcwFn<X, G>,
        mut combine: impl FnMut(F, F) -> F,
    ) -> PcwOut
    where
        F: Clone,
        PcwOut: PcwFn<X, F>,
    {
        match (self.segment_count(), other.segment_count()) {
            (0, _) => panic!("Empty function is invalid"),
            (_, 0) => panic!("Empty function is invalid"),
            (1, n) => PcwOut::try_from_iters(
                other.into_jumps(),
                iter::repeat(self.into_funcs().next().unwrap()).take(n),
            )
            .unwrap(),
            (_, 1) => PcwOut::try_from_iters(
                other.into_jumps(),
                iter::once(self.into_funcs().reduce(combine).unwrap()),
            )
            .unwrap(),
            (_, n) => {
                let (jl, mut fl) = self.into_jumps_and_funcs();
                let mut funcs = Vec::with_capacity(n);
                let mut active_f = fl.next().unwrap();
                funcs.push(active_f.clone()); // value of result sufficiently far left in the domain
                for c in jl.merge_join_by(other.jumps(), |x, y| x.cmp(y)) {
                    match c {
                        EitherOrBoth::Left(_) => {
                            if let Some(new_f) = fl.next() {
                                active_f = new_f.clone();
                                let f = combine(funcs.pop().unwrap(), new_f);
                                funcs.push(f);
                            }
                        }
                        EitherOrBoth::Right(_) => funcs.push(active_f.clone()),
                        EitherOrBoth::Both(_, _) => {
                            if let Some(new_f) = fl.next() {
                                active_f = new_f.clone();
                                funcs.push(new_f);
                            } else {
                                funcs.push(active_f.clone())
                            }
                        }
                    }
                }
                PcwOut::try_from_iters(other.into_jumps(), funcs).unwrap()
            }
        }
    }

    /// Find the function that locally defines the piecewise function at some point `x` of
    /// the domain.
    fn func_at(&self, x: &X) -> &F {
        match self.segment_count() {
            0 => panic!("Empty function is invalid"),
            1 => &self.funcs()[0],
            _ => match self.jumps().binary_search(x) {
                Ok(jump_idx) => &self.funcs()[jump_idx + 1],
                Err(insertion_idx) => &self.funcs()[insertion_idx],
            },
        }
    }

    /// Find the function that locally defines the piecewise function at some point `x` of
    /// the domain.
    fn func_at_mut(&mut self, x: &X) -> &mut F {
        match self.segment_count() {
            0 => panic!("Empty function is invalid"),
            1 => &mut self.funcs_mut()[0],
            _ => match self.jumps().binary_search(x) {
                Ok(jump_idx) => &mut self.funcs_mut()[jump_idx + 1],
                Err(insertion_idx) => &mut self.funcs_mut()[insertion_idx],
            },
        }
    }

    /// Evaluate the function at some point `x` of the domain.
    fn eval<Y>(&self, x: X) -> Y
    where
        F: Fn(X) -> Y,
    {
        self.func_at(&x)(x)
    }

    /// Mutably evaluate the function at some point `x` of the domain.
    fn eval_mut<Y>(&mut self, x: X) -> Y
    where
        F: FnMut(X) -> Y,
    {
        self.func_at_mut(&x)(x)
    }
}

/// A piecewise function internally backed by `Vec`s
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct VecPcwFn<X, F> {
    jumps: Vec<X>,
    funcs: Vec<F>,
}

impl<X, F> Kind1To1 for VecPcwFn<X, F> {
    type Constructor<S> = VecPcwFn<X, S>;
}

impl<X, F> Functor<F> for VecPcwFn<X, F> {
    fn fmap<S>(self, f: impl FnMut(F) -> S) -> Self::Constructor<S> {
        VecPcwFn {
            jumps: self.jumps,
            funcs: self.funcs.into_iter().map(f).collect(),
        }
    }
}

impl<X, F> FunctorRef<F> for VecPcwFn<X, F>
where
    X: Clone,
{
    fn fmap_ref<S>(&self, f: impl FnMut(&F) -> S) -> Self::Constructor<S> {
        VecPcwFn {
            jumps: self.jumps.clone(),
            funcs: self.funcs.iter().map(f).collect(),
        }
    }
}

fn strictly_less<T: PartialOrd>(x: &T, y: &T) -> Option<Ordering> {
    use Ordering::*;
    match x.partial_cmp(y) {
        Some(Less) => Some(Less),
        _ => None,
    }
}

impl<X: Ord, F> PcwFn<X, F> for VecPcwFn<X, F> {
    type JmpIter = <Vec<X> as IntoIterator>::IntoIter;
    type FncIter = <Vec<F> as IntoIterator>::IntoIter;

    fn jumps(&self) -> &[X] {
        &self.jumps
    }

    fn funcs(&self) -> &[F] {
        &self.funcs
    }

    fn funcs_mut(&mut self) -> &mut [F] {
        &mut self.funcs
    }

    fn try_from_iters<Jmp: IntoIterator<Item = X>, Fnc: IntoIterator<Item = F>>(
        jumps: Jmp,
        funcs: Fnc,
    ) -> Result<Self, PcwFnError> {
        use std::cmp::Ordering::*;
        let jumps = jumps.into_iter().collect_vec();
        let funcs = funcs.into_iter().collect_vec();
        //if !jumps.iter().is_strictly_sorted() {
        if !IsSorted::is_sorted_by(&mut jumps.iter(), strictly_less) {
            Err(PcwFnError::JumpsNotStrictlySorted)
        } else {
            match (jumps.iter().len() + 1).cmp(&funcs.iter().len()) {
                Greater => Err(PcwFnError::TooManyJumpsForFuncs),
                Less => Err(PcwFnError::TooManyJumpsForFuncs),
                Equal => Ok(VecPcwFn { jumps, funcs }),
            }
        }
    }

    fn add_segment(&mut self, jump: X, func: F) {
        self.jumps.push(jump);
        self.funcs.push(func);
    }

    fn into_jumps_and_funcs(self) -> (Self::JmpIter, Self::FncIter) {
        (self.jumps.into_iter(), self.funcs.into_iter())
    }
}

pub use num_impls::*;
mod num_impls {
    use super::*;
    use num_traits::{One, Pow, Zero};
    use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

    /// Lifts a basic binary operation from functions to piecewise functions.
    macro_rules! pointwise_owned_binop_impl {
        ( $trait_to_impl:ident, $method_name:ident, $for_type:ident ) => {
            impl<Rhs, X, F> $trait_to_impl<Rhs> for $for_type<X, F>
            where
                X: Ord,
                Rhs: PcwFn<X, F>,
                F: $trait_to_impl<F> + Clone,
            {
                type Output = VecPcwFn<X, F::Output>;
                fn $method_name(self, rhs: Rhs) -> Self::Output {
                    self.combine(rhs, $trait_to_impl::$method_name)
                }
            }
        };
    }

    /* the above macro produces impls like

    impl<R, X, F> Add<R> for VecPcwFn<X, F>
    where
        X: Ord,
        R: PcwFn<X, F>,
        F: Add<F> + Clone,
    {
        type Output = VecPcwFn<X, F::Output>;

        fn add(self, rhs: R) -> Self::Output {
            self.combine(rhs, Add::add)
        }
    }

    */

    pointwise_owned_binop_impl!(Add, add, VecPcwFn);
    pointwise_owned_binop_impl!(Sub, sub, VecPcwFn);
    pointwise_owned_binop_impl!(Mul, mul, VecPcwFn);
    pointwise_owned_binop_impl!(Div, div, VecPcwFn);
    pointwise_owned_binop_impl!(Pow, pow, VecPcwFn);
    pointwise_owned_binop_impl!(Rem, rem, VecPcwFn);
    pointwise_owned_binop_impl!(BitAnd, bitand, VecPcwFn);
    pointwise_owned_binop_impl!(BitOr, bitor, VecPcwFn);
    pointwise_owned_binop_impl!(BitXor, bitxor, VecPcwFn);
    pointwise_owned_binop_impl!(Shl, shl, VecPcwFn);
    pointwise_owned_binop_impl!(Shr, shr, VecPcwFn);

    impl<X, F> Zero for VecPcwFn<X, F>
    where
        X: Ord,
        F: Zero + Clone,
    {
        fn zero() -> Self {
            Self::global(F::zero())
        }

        fn is_zero(&self) -> bool {
            self.segment_count() == 1 && self.funcs[0].is_zero()
        }
    }

    impl<X, F> One for VecPcwFn<X, F>
    where
        X: Ord,
        F: One + Clone + PartialEq,
    {
        fn one() -> Self {
            Self::global(F::one())
        }
        fn is_one(&self) -> bool
        where
            Self: PartialEq,
        {
            self.segment_count() == 1 && self.funcs[0].is_one()
        }
    }

    /// Lifts a basic unary operation from functions to piecewise functions
    macro_rules! pointwise_owned_unop_impl {
        ( $trait_to_impl:ident, $method_name:ident, $for_type:ident ) => {
            impl<X, F> $trait_to_impl for $for_type<X, F>
            where
                X: Ord,
                F: $trait_to_impl,
            {
                type Output = VecPcwFn<X, F::Output>;
                fn $method_name(self) -> Self::Output {
                    self.fmap($trait_to_impl::$method_name)
                }
            }
        };
    }

    pointwise_owned_unop_impl!(Not, not, VecPcwFn);
    pointwise_owned_unop_impl!(Neg, neg, VecPcwFn);
}

#[cfg(test)]
mod tests {
    use super::*;

    mod add {
        use super::*;

        #[test]
        fn same_domains() {
            let f = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![2, 4, 6, 8]).unwrap();
            assert_eq!(
                f + g,
                VecPcwFn::try_from_iters(vec!(5, 10, 15), vec![3, 6, 9, 12]).unwrap()
            )
        }

        #[test]
        fn left_domain_larger() {
            let f = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![10], vec![4, 6]).unwrap();
            assert_eq!(
                VecPcwFn::try_from_iters(vec!(5, 10, 15), vec![5, 6, 9, 10]).unwrap(),
                f + g
            )
        }

        #[test]
        fn right_domain_larger() {
            let g = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let f = VecPcwFn::try_from_iters(vec![10], vec![4, 6]).unwrap();
            assert_eq!(
                VecPcwFn::try_from_iters(vec!(5, 10, 15), vec![5, 6, 9, 10]).unwrap(),
                f + g
            )
        }

        #[test]
        fn unaligned_domains() {
            let f = VecPcwFn::try_from_iters(vec![0], vec![1, 3]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![1], vec![2, 4]).unwrap();
            assert_eq!(
                VecPcwFn::try_from_iters(vec!(0, 1), vec![3, 5, 7]).unwrap(),
                f + g
            );
            // same thing, just a bigger example
            let f = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![7, 12], vec![4, 6, -5]).unwrap();
            assert_eq!(
                VecPcwFn::try_from_iters(vec!(5, 7, 10, 12, 15), vec![5, 6, 8, 9, -2, -1]).unwrap(),
                f + g,
            )
        }
    }

    mod resample {
        use super::*;
        #[test]
        fn same_domains() {
            let f = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![2, 4, 6, 8]).unwrap();
            let h: VecPcwFn<_, _> = f.resample_to(g, std::cmp::min);
            assert_eq!(
                h,
                VecPcwFn::try_from_iters(vec!(5, 10, 15), vec![1, 2, 3, 4]).unwrap()
            )
        }

        #[test]
        fn left_domain_larger() {
            let f = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![10], vec![4, 6]).unwrap();
            let h: VecPcwFn<_, _> = f.resample_to(g, std::cmp::min);
            assert_eq!(h, VecPcwFn::try_from_iters(vec!(10), vec![1, 3]).unwrap())
        }

        #[test]
        fn right_domain_larger() {
            let f = VecPcwFn::try_from_iters(vec![10], vec![4, 6]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let h: VecPcwFn<_, _> = f.resample_to(g, std::cmp::min);
            assert_eq!(
                h,
                VecPcwFn::try_from_iters(vec!(5, 10, 15), vec![4, 4, 6, 6]).unwrap()
            )
        }

        #[test]
        fn unaligned_domains() {
            let f = VecPcwFn::try_from_iters(vec![0], vec![1, 3]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![1], vec![2, 4]).unwrap();
            let h: VecPcwFn<_, _> = f.resample_to(g, std::cmp::min);
            assert_eq!(h, VecPcwFn::try_from_iters(vec!(1), vec![1, 3]).unwrap());
            // same thing, just a bigger example
            let f = VecPcwFn::try_from_iters(vec![5, 10, 15], vec![1, 2, 3, 4]).unwrap();
            let g = VecPcwFn::try_from_iters(vec![7, 12], vec![4, 6, -5]).unwrap();
            let h: VecPcwFn<_, _> = f.resample_to(g, std::cmp::min);
            assert_eq!(
                h,
                VecPcwFn::try_from_iters(vec!(7, 12), vec![1, 2, 3]).unwrap()
            )
        }

        #[test]
        fn unaligned_domains_big() {
            let f = VecPcwFn::try_from_iters(
                vec![5, 10, 15, 16, 17, 20],
                vec![2, 1, 6, 4, 5, -10, 100],
            )
            .unwrap();
            let g = VecPcwFn::try_from_iters(vec![7, 12, 30], vec![4, 6, -5, -20]).unwrap();
            let h: VecPcwFn<_, _> = f.resample_to(g, std::cmp::min);
            assert_eq!(
                h,
                VecPcwFn::try_from_iters(vec!(7, 12, 30), vec![1, 1, -10, 100]).unwrap()
            )
        }
    }

    #[test]
    fn eval() {
        let f: VecPcwFn<_, &dyn Fn(i32) -> i32> = VecPcwFn::try_from_iters(
            vec![5, 10, 15],
            vec![
                &(|_| -10) as &dyn Fn(i32) -> i32,
                &(|_| 10) as &dyn Fn(i32) -> i32,
                &(|_| 5) as &dyn Fn(i32) -> i32,
                &(|_| -5) as &dyn Fn(i32) -> i32,
            ],
        )
        .unwrap();
        assert_eq!(f.eval(0), -10);
        assert_eq!(f.eval(4), -10);
        assert_eq!(f.eval(5), 10);
        assert_eq!(f.eval(6), 10);
        assert_eq!(f.eval(10), 5);
        assert_eq!(f.eval(200), -5);
    }
}
