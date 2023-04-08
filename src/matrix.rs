pub use std::ops::{Add, Sub, Mul};
use crate::num::*;

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    elem: Vec<Vec<T>>
}

impl<T> Matrix<T> {
    pub fn from_vec(v: Vec<Vec<T>>) -> Self {
        if ! v.is_empty() {
            let n = v[0].len();
            assert!(v.iter().all(|row| row.len() == n))
        }

        Self { elem: v }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self.elem.len() {
            0 => (0,0),
            m => (m, self.elem[0].len())
        }
    }

    fn is_square(&self) -> bool {
        let (m,n) = self.shape();
        m == n
    }
}

impl<T: Num> Matrix<T> {
    fn zeros(m: usize, n: usize) -> Self {
        Self { elem: vec![vec![T::zero(); n]; m]}
    }

    pub fn e(n: usize) -> Self {
        let mut x = Self::zeros(n, n);
        for i in 0..n {
            x.elem[i][i] = T::one();
        }
        x
    }

    // transposed
    pub fn t(&self) -> Self {
        let (m, n) = self.shape();
        let mut transposed = Self::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                transposed.elem[i][j] = self.elem[j][i];
            }
        }
        transposed
    }

    pub fn scalar_mul(&self, k: T) -> Self {
        map_elem(|x| x * k, self)
    }

    pub fn pow(self, exp: u32) -> Self {
        assert!(self.is_square());

        match exp {
            0 => panic!("pow by 0 not allowed"),
            1 => self,
            x => self.clone() * self.pow(x-1)
        }
    }
}

impl<T: Num> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (m,_n) = self.shape();
        for i in 0..m {
            let si = self.elem[i].iter().map(|x| x.to_string()).reduce(|mut acc, x| {
                acc.push(' ');
                acc.push_str(&x);
                acc
            }).unwrap();
            if i == m-1 {
                return write!(f, "{}", si);
            } else {
                writeln!(f, "{}", si)?;
            }
        }
        Ok(())
    }
}

impl<T: Num> Add for Matrix<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl<T: Num> Add for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        zip_with(|x, y| x + y, &self, &rhs)
    }
}

impl<T: Num> Sub for Matrix<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}

impl<T: Num> Sub for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        zip_with(|x, y| x - y, self, rhs)
    }
}

impl<T: Num> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<T: Num> Mul for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let (l, m) = self.shape();
        let (m2, n) = rhs.shape();

        assert_eq!(m, m2);

        let mut matrix = Self::Output::zeros(l, n);
        for i in 0..l {
            for j in 0..n {
                matrix.elem[i][j] =
                    (0..m)
                    .map(|h| self.elem[i][h] * rhs.elem[h][j])
                    .sum();
            }
        }
        matrix
    }
}

impl<T: RealNum> Matrix<Complex<T>> {
    pub fn c(&self) -> Self {
        map_elem(|y| y.c(), self)
    }

    pub fn hermite(&self) -> Self {
        self.t().c()
    }
}

fn zip_with<F, T: Num, U: Num>(f: F, m1: &Matrix<T>, m2: &Matrix<T>) -> Matrix<U>
where F: Fn(T, T) -> U {
    assert_eq!(m1.shape(), m2.shape());

    let v =
    m1.elem.iter().zip(m2.elem.iter()).map(
        |(v1, v2)| v1.iter().zip(v2.iter()).map(
            |(e1, e2)| f(*e1, *e2)
        ).collect()
    ).collect();
    Matrix::from_vec(v)
}

fn map_elem<F, T: Num, U: Num>(f: F, x: &Matrix<T>) -> Matrix<U>
where F: Fn(T) -> U {
    let v =
    x.elem.iter().map(
        |row| row.iter().map(
            |e| f(*e)
        ).collect()
    ).collect();
    Matrix::from_vec(v)
}
