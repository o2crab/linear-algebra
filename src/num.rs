pub mod complex;

use std::ops::{Add, Sub, Mul, Neg};
use std::fmt::Display;
use std::iter::Sum;


pub trait Num : Clone + Copy + Display + PartialEq + Neg<Output = Self> + Add<Self, Output = Self> + Zero + Sub<Self, Output = Self> + Mul<Self, Output = Self> + Sum {
}

impl Num for i32 {
}

impl Num for f32 {
}


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
