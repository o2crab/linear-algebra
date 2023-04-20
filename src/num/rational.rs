use super::*;
use std::ops::{Rem, Div};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rational<T>(T, T);

impl<T: Integer> Rational<T> {
    pub fn new(a: T, b: T) -> Self {
        assert!(
            b != T::zero(),
            "denominator cannot be 0"
        );

        // make the denominator positive
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

    pub fn inverse(self) -> Option<Self> {
        if self.0 == T::zero() {
            return None;
        }

        Some(Self::new(self.1, self.0))
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

impl<T: Integer> ExactNum for Rational<T> {
    type DivOutput = Rational<T>;

    fn div(self, rhs: Self) -> Self::DivOutput {
        assert_ne!(
            rhs, Self::zero(),
            "cannot divide by 0"
        );

        self * rhs.inverse().unwrap()
    }
}


pub trait Integer: Num + Div<Output = Self> + Rem<Output = Self> + PartialOrd {}

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

impl<T: Integer> ExactNum for T {
    type DivOutput = Rational<T>;

    fn div(self, rhs: Self) -> Self::DivOutput {
        assert_ne!(
            rhs, Self::zero(),
            "cannot divide by 0"
        );

        Rational::new(self, rhs)
    }
}

impl Integer for i32 {}




#[cfg(test)]
mod tests {
    use super::*;

    mod gcd {
        use super::*;

        #[test]
        fn gcd_18_12_return_6() {
            assert_eq!(
                gcd(18, 12),
                6
            )
        }

        #[test]
        fn gcd_12_18_return_6() {
            assert_eq!(
                gcd(12, 18),
                6
            )
        }

        #[test]
        fn gcd_0_b_return_b() {
            let b = 24;
            assert_eq!(
                gcd(0, b),
                b
            )
        }

        #[test]
        fn gcd_neg_b_return_gcd_pos_b() {
            let a = -6;
            let b = 9;
            assert_eq!(
                gcd(a, b),
                gcd(-a, b)
            )
        }

        #[test]
        fn gcd_a_neg_return_gcd_a_pos() {
            let a = 6;
            let b = -9;
            assert_eq!(
                gcd(a, b),
                gcd(a, -b)
            )
        }
    }

    mod abs {
        use super::*;

        #[test]
        fn pos_return_pos() {
            let x = 2;
            assert_eq!(
                abs(x),
                x
            )
        }

        #[test]
        fn zero_return_zero() {
            let x = 0;
            assert_eq!(
                abs(x),
                x
            )
        }

        #[test]
        fn neg_return_pos() {
            let x = -2;
            assert_eq!(
                abs(x),
                -x
            )
        }
    }

    mod new {
        use super::*;

        #[test]
        #[should_panic(expected = "denominator cannot be 0")]
        fn denominator_cannot_be_0() {
            Rational::new(2, 0);
        }

        #[test]
        fn reduced_return_reduced() {
            assert_eq!(
                Rational::new(2, 3),
                Rational(2, 3)
            )
        }

        #[test]
        fn unreduced_return_reduced() {
            assert_eq!(
                Rational::new(12, 18),
                Rational(2, 3)
            )
        }

        #[test]
        fn zero_return_rational_0_1() {
            assert_eq!(
                Rational::new(0, -6),
                Rational(0, 1)
            )
        }

        #[test]
        fn denominator_become_positive() {
            assert_eq!(
                Rational::new(12, -18),
                Rational(-2, 3)
            )
        }
    }

    mod reduce {
        use super::*;

        #[test]
        fn reduced_return_itself() {
            let x = Rational(2, 3);
            assert_eq!(
                x.reduce(),
                Rational(2, 3)
            )
        }

        #[test]
        fn unreduced_return_reduced() {
            let x = Rational(12, 18);
            assert_eq!(
                x.reduce(),
                Rational(2, 3)
            )
        }
    }

    #[test]
    fn add_1_3_and_1_6_return_1_2() {
        assert_eq!(
            Rational::new(1,3) + Rational::new(1,6),
            Rational::new(1,2)
        )
    }

    #[test]
    fn sub() {
        assert_eq!(
            Rational::new(1,6) - Rational::new(1,2),
            Rational::new(-1,3)
        )
    }

    #[test]
    fn mul() {
        assert_eq!(
            Rational::new(-3, 8) * Rational::new(10, 9),
            Rational::new(-5, 12)
        )
    }

    mod div {
        use super::*;

        #[test]
        #[should_panic(expected = "cannot divide by 0")]
        fn div_by_0_panic() {
            let _ = Rational::new(5, 3).div(Rational::zero());
        }

        #[test]
        fn div_return_inverse_mul() {
            let r1 = Rational::new(5, 3);
            let r2 = Rational::new(7, 6);
            assert_eq!(
                r1.div(r2),
                Rational::new(10, 7)
            );
        }
    }

    mod integer_div {
        use super::*;

        #[test]
        #[should_panic(expected = "cannot divide by 0")]
        fn div_by_0_panic() {
            let _ = ExactNum::div(3, 0);
        }

        #[test]
        fn div_return_rational() {
            assert_eq!(
                ExactNum::div(9, 42),
                Rational::new(3, 14)
            );
        }
    }
}
