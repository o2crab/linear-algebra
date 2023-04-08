use super::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Complex<T>(pub T, pub T);

impl<T: Num> Complex<T> {
    pub fn re(&self) -> T {
        self.0
    }

    pub fn im(&self) -> T {
        self.1
    }

    pub fn c(&self) -> Self {
        Self(self.re(), - self.im())
    }
}

impl<T: Num> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im() == T::zero() {
            return write!(f, "{}", self.re());
        }
        
        if self.re() == T::zero() {
            let im =
                match self.im() {
                    x if x == T::one() => String::new(),
                    x if x == - T::one() => String::from("-"),
                    x => x.to_string()
                };
            return write!(f, "{}i", im);
        }

        let im =
            match self.im() {
                x if x == T::one() => String::from("+"),
                x if x == - T::one() => String::from("-"),
                x => format!("{:+}", x)
            };
        write!(f, "{}{:+}i", self.re(), im)
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
            self.re() * rhs.re() - self.im() * rhs.im(),
            self.re() * rhs.im() + self.im() * rhs.re()
        )
    }
}

impl<T: Num> One for Complex<T> {
    fn one() -> Self {
        Self(T::one(), T::one())
    }
}

impl<T: Num> Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Self::add)
    }
}

impl<T: RealNum> Num for Complex<T> {}
