pub use std::ops::{Add, Sub, Mul};
// use num::{Num};

#[derive(Clone)]
pub struct Matrix {
    pub elem: Vec<Vec<i32>>
}

impl Matrix {
    pub fn from_vec(v: Vec<Vec<i32>>) -> Self {
        Self { elem: v }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self.elem.len() {
            0 => (0,0),
            m => (m, self.elem[0].len())
        }
    }

    fn is_square(&self) -> bool {
        let (m, n) = self.shape();
        m == n
    }

    fn zeros(m: usize, n: usize) -> Self {
        Self { elem: vec![vec![0; n]; m]}
    }

    pub fn scalar_mul(&self, k: i32) -> Self {
        map_elem(|x| x * k, self)
    }

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

    pub fn pow(&self, n: usize) -> Self {
        assert!(self.is_square());

        match n {
            1 => self.clone(),
            k => self * &self.pow(k-1)
        }
    }
}

impl std::fmt::Display for Matrix {
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

impl Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        zip_with(|x, y| x + y, &self, &rhs)
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        zip_with(|x, y| x - y, self, rhs)
    }
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        let (l, m) = self.shape();
        let (m2, n) = rhs.shape();

        assert_eq!(m, m2);

        let mut matrix = Matrix::zeros(l, n);
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

fn zip_with<F>(f: F, m1: &Matrix, m2: &Matrix) -> Matrix
where F: Fn(i32, i32) -> i32 {
    let v =
    m1.elem.iter().zip(m2.elem.iter()).map(
        |(v1, v2)| v1.iter().zip(v2.iter()).map(
            |(e1, e2)| f(*e1, *e2)
        ).collect()
    ).collect();
    Matrix::from_vec(v)
}

fn map_elem<F>(f: F, x: &Matrix) -> Matrix
where F: Fn(i32) -> i32 {
    let v =
    x.elem.iter().map(
        |row| row.iter().map(
            |e| f(*e)
        ).collect()
    ).collect();
    Matrix::from_vec(v)
}
