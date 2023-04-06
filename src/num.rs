pub mod complex;

use std::ops::{Add, Sub, Mul, Neg};
use std::fmt::Display;
use std::iter::Sum;


pub trait Num : Clone + Copy + Display + PartialEq + Neg<Output = Self> + Add<Self, Output = Self> + Zero + Sub<Self, Output = Self> + Mul<Self, Output = Self> + Sum {
}


trait RealNum: Clone + Copy + Display + PartialEq + Neg<Output = Self> + Add<Self, Output = Self> + Zero + Sub<Self, Output = Self> + Mul<Self, Output = Self> + Sum {}

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
