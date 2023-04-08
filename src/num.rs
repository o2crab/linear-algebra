mod complex;

use std::ops::{Add, Sub, Mul, Neg};
use std::fmt::Display;
use std::iter::Sum;
pub use complex::Complex;


pub trait Num : Clone + Copy + Display + PartialEq + Neg<Output = Self> + Add<Self, Output = Self> + Zero + Sub<Self, Output = Self> + Mul<Self, Output = Self> + One + Sum {
}


pub trait RealNum: Clone + Copy + Display + PartialEq + Neg<Output = Self> + Add<Self, Output = Self> + Zero + Sub<Self, Output = Self> + Mul<Self, Output = Self> + One + Sum {}

impl RealNum for i32 {}
impl RealNum for f32 {}

impl<T: RealNum> Num for T {}


pub trait Zero {
    fn zero() -> Self;
}

impl Zero for i32 {
    fn zero() -> Self {
        0
    }
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}


pub trait One {
    fn one() -> Self;
}

impl One for i32 {
    fn one() -> Self {
        1
    }
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}