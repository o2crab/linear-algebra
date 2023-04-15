use super::*;
use std::ops::{Rem, Div};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rational<T>(T, T);

impl<T: Integer> Rational<T> {
    pub fn new(a: T, b: T) -> Self {
        assert!(b != T::zero());

        if b < T::zero() {
            Self(-a, -b).reduce()
        } else {
            Self(a, b).reduce()
        }
    }

    fn reduce(self) -> Self {
        let d = gcd(self.0, self.1);
        let numerator = self.0 / d;
        let denominator = self.1 / d;
        Self(numerator, denominator)
    }
}

impl<T: Integer> Display for Rational<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.0, self.1)
    }
}

impl<T: Integer> Neg for Rational<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0, self.1)
    }
}

impl<T: Integer> Add for Rational<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 * rhs.1 + rhs.0 * self.1, self.1 * rhs.1)
    }
}

impl<T: Integer> Zero for Rational<T> {
    fn zero() -> Self {
        Self(T::zero(), T::one())
    }
}

impl<T: Integer> Sub for Rational<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.0 * rhs.1 - rhs.0 * self.1, self.1 * rhs.1)
    }
}

impl<T: Integer> Mul for Rational<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl<T: Integer> One for Rational<T> {
    fn one() -> Self {
        Self(T::one(), T::one())
    }
}

impl<T: Integer> Sum for Rational<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Self::add)
    }
}

impl<T: Integer> RealNum for Rational<T> {}

fn gcd<T: Integer>(a: T, b: T) -> T {
    let a = abs(a);
    let b = abs(b);
    if b == T::zero() {
        a
    } else {
        gcd(b, a % b)
    }
}

fn abs<T: Integer>(x: T) -> T {
    if x < T::zero() {
        -x
    } else {
        x
    }
}

pub trait Integer: Num + Div<Output = Self> + Rem<Output = Self> + PartialOrd {}

impl Integer for i32 {}
