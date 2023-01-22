//! Some FP traits.

use itertools::Itertools;

/// A trait for types of kind * -> *
pub trait Kind1To1 {
    type Constructor<S>;
    // where
    //     Self::Constructor<S>: Kind1To1; // : ?Sized
}

/// A trait for types that can be mapped over to obtain an instance of "the same type" with a
/// different type parameter.
pub trait Functor<T>: Kind1To1 {
    fn fmap<S>(self, f: impl FnMut(T) -> S) -> Self::Constructor<S>;
    // where
    //     Self::Constructor<S>: Kind1To1;
    // where
    //     Self::Constructor<S>: Functor<S>;
}

/// A trait for types that can be mapped over by reference.
pub trait FunctorRef<T>: Kind1To1 {
    fn fmap_ref<S>(&self, f: impl FnMut(&T) -> S) -> Self::Constructor<S>;
    // where
    //     Self::Constructor<S>: Kind1To1;
    // where
    //     Self::Constructor<S>: FunctorRef<S>;
}

impl<T> Kind1To1 for Vec<T> {
    type Constructor<S> = Vec<S>;
}

impl<T> Functor<T> for Vec<T> {
    fn fmap<S>(self, f: impl FnMut(T) -> S) -> Vec<S> {
        self.into_iter().map(f).collect()
    }
}

impl<T> FunctorRef<T> for Vec<T> {
    fn fmap_ref<S>(&self, f: impl FnMut(&T) -> S) -> Vec<S> {
        self.iter().map(f).collect()
    }
}

impl<T, const N: usize> Kind1To1 for [T; N] {
    type Constructor<S> = [S; N];
}

impl<T, const N: usize> Functor<T> for [T; N] {
    fn fmap<S>(self, f: impl FnMut(T) -> S) -> [S; N] {
        self.map(f)
    }
}

impl<T, const N: usize> FunctorRef<T> for [T; N] {
    fn fmap_ref<S>(&self, f: impl FnMut(&T) -> S) -> [S; N] {
        unsafe {
            self.iter()
                .map(f)
                .collect_vec() // this really isn't great but apparently there's not really a better way to do this currently
                .try_into()
                .unwrap_unchecked()
        }
    }
}

impl<T> Kind1To1 for Box<[T]> {
    type Constructor<S> = Box<[S]>;
}

impl<T> Functor<T> for Box<[T]> {
    fn fmap<S>(self, f: impl FnMut(T) -> S) -> Box<[S]> {
        Vec::from(self).fmap(f).into_boxed_slice()
    }
}

impl<T> FunctorRef<T> for Box<[T]> {
    fn fmap_ref<S>(&self, f: impl FnMut(&T) -> S) -> Box<[S]> {
        self.iter().map(f).collect_vec().into_boxed_slice()
    }
}
// Only implementable if `Constructor<S>` is `?Sized` in trait def
// impl<T> Kind1To1 for [T] {
//     type Constructor<S> = [S];
// }
