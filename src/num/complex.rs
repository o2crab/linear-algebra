use super::*;

#[derive(Clone, Copy, PartialEq)]
pub struct Complex<T>(pub T, pub T);

impl<T: Num> Complex<T> {
    pub fn re(&self) -> T {
        self.0
    }

    pub fn im(&self) -> T {
        self.1
    }

    pub fn conjugate(&self) -> Self {
        Self(self.re(), - self.im())
    }
}
impl<T: Num> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.im() {
            x if x == T::zero() => write!(f, "{}", self.re()),
            _ => write!(f, "{}{:+}i", self.re(), self.im())
        }
    }
}

impl<T: Num> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.re(), -self.im())
    }
}

impl<T: Num> Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(
            self.re() + rhs.re(),
            self.im() + rhs.im()
        )
    }
}

impl<T: Num> Zero for Complex<T> {
    fn zero() -> Self {
        Self(T::zero(), T::zero())
    }
}

impl<T: Num> Sub for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(
            self.re() - rhs.re(),
            self.im() - rhs.im()
        )
    }
}

impl<T: Num> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(
            self.re() * rhs.re(),
            self.im() * rhs.im()
        )
    }
}

impl<T: Num> Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Self::add)
    }
}

impl<T: RealNum> Num for Complex<T> {}
