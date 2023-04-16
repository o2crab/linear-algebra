use super::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Complex<T>(T, T);

impl<T: RealNum> Complex<T> {
    pub fn new(re: T, im: T) -> Self {
        Self(re, im)
    }
}

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



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_return_real_part() {
        let c = Complex(1, 2);
        assert_eq!(
            c.re(),
            1
        )
    }

    #[test]
    fn im_return_imaginary_part() {
        let c = Complex(1, 2);
        assert_eq!(
            c.im(),
            2
        )
    }

    #[test]
    fn c_return_conjugate() {
        let c = Complex(1, 2);
        assert_eq!(
            c.c(),
            Complex(1, -2)
        )
    }

    #[test]
    fn neg_reverse_re_and_im() {
        let c = Complex(1, 2);
        assert_eq!(
            -c,
            Complex(-1, -2)
        )
    }

    #[test]
    fn add() {
        let c1 = Complex(1, 2);
        let c2 = Complex(2, -3);
        assert_eq!(
            c1 + c2,
            Complex(3, -1)
        )
    }

    #[test]
    fn sub() {
        let c1 = Complex(1, 2);
        let c2 = Complex(2, -3);
        assert_eq!(
            c1 - c2,
            Complex(-1, 5)
        )
    }

    #[test]
    fn mul_re_return_re_prod() {
        let c1 = Complex(2, 0);
        let c2 = Complex(3, 0);
        assert_eq!(
            c1 * c2,
            Complex(6, 0)
        )
    }

    #[test]
    fn mul_im_return_neg_re_prod() {
        let c1 = Complex(0, 2);
        let c2 = Complex(0, 3);
        assert_eq!(
            c1 * c2,
            Complex(-6, 0)
        )
    }

    #[test]
    fn mul_mix() {
        let c1 = Complex(1, 2);
        let c2 = Complex(3, 4);
        assert_eq!(
            c1 * c2,
            Complex(1*3 - 2*4, 1*4 + 2*3)
        )
    }
}
