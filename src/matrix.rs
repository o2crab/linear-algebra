pub use std::ops::{Add, Sub, Mul};
use crate::num::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    // has at least one element
    // every row has the same number of elements
    elem: Vec<Vec<T>>
}

impl<T> Matrix<T> {
    pub fn from_vec(v: Vec<Vec<T>>) -> Self {
        assert!(
            !v.is_empty() && !v[0].is_empty(),
            "cannot create an empty matrix"
        );
        let n = v[0].len();
        assert!(
            v.iter().all(|row| row.len() == n),
            "every row of a matrix must have the same length"
        );

        Self { elem: v }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.elem.len(), self.elem[0].len())
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

    pub fn pow(&self, exp: u32) -> Self {
        assert!(
            self.is_square(),
            "only square matrices can be powered"
        );

        match exp {
            0 => panic!("pow by 0 not allowed"),
            1 => self.clone(),
            x if x % 2 == 0 => (self * self).pow(x / 2),
            x => &(self * self).pow(x / 2) * self
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



#[cfg(test)]
mod tests {
    use super::*;

    mod from_vec {
        use super::*;

        #[test]
        #[should_panic(expected = "cannot create an empty matrix")]
        fn cannot_create_empty_matrix() {
            let _mat: Matrix<i32> = Matrix::from_vec(Vec::new());
        }

        #[test]
        #[should_panic(expected = "every row of a matrix must have the same length")]
        fn different_row_length_panic() {
            let _mat = Matrix::from_vec(vec![
                vec![1, 2],
                vec![1, 2, 3]
            ]);
        }

        #[test]
        fn create_i32_matrix() {
            let result = Matrix::from_vec(vec![
                vec![1, -15],
                vec![153, 0],
                vec![0, 0]
            ]);
            assert_eq!(
                result,
                Matrix { elem: vec![
                    vec![1, -15],
                    vec![153, 0],
                    vec![0, 0]
                ]}
            );
        }

        #[test]
        fn create_f32_matrix() {
            let result = Matrix::from_vec(vec![
                vec![1.5, -15.2],
                vec![0.004, 243.77],
                vec![0.1, -0.011]
            ]);
            assert_eq!(
                result,
                Matrix { elem: vec![
                    vec![1.5, -15.2],
                    vec![0.004, 243.77],
                    vec![0.1, -0.011]
                ]}
            );
        }
    }

    mod shape {
        use super::*;

        #[test]
        fn matrix_2_3_return_3_2() {
            let matrix_2_3 = Matrix{ elem: vec![
                vec![1, 2, 3],
                vec![1, 3, 2],
            ]};
            assert_eq!(
                matrix_2_3.shape(),
                (2, 3)
            )
        }
    }

    mod is_square {
        use super::*;

        #[test]
        fn matrix_2_2_is_square() {
            let mat_2_2 = Matrix{ elem: vec![
                vec![1, 2],
                vec![4, -2]
            ]};
            assert!(mat_2_2.is_square());
        }

        #[test]
        fn matrix_1_2_is_not_square() {
            let mat_1_2 = Matrix { elem: vec![
                vec![1, 2]
            ]};
            assert!(!mat_1_2.is_square());
        }
    }

    mod zero {
        use super::*;

        #[test]
        fn create_matrix_2_3() {
            let mat: Matrix<i32> = Matrix::zeros(2, 3);
            assert_eq!(
                mat,
                Matrix{ elem: vec![ vec![0; 3]; 2] }
            );
        }
    }

    mod e {
        use super::*;

        #[test]
        fn n1_create_e1() {
            let mat: Matrix<i32> = Matrix::e(1);
            assert_eq!(
                mat,
                Matrix{ elem: vec![vec![1]] }
            );
        }

        #[test]
        fn n3_create_e3() {
            let mat: Matrix<i32> = Matrix::e(3);
            assert_eq!(
                mat,
                Matrix{ elem: vec![
                    vec![1, 0, 0],
                    vec![0, 1, 0],
                    vec![0, 0, 1]
                ]}
            );
        }
    }

    mod t {
        use super::*;

        #[test]
        fn transpose_matrix_2_3() {
            let mat = Matrix{ elem: vec![
                vec![11, 12],
                vec![21, 22],
                vec![31, 32]
            ]};
            assert_eq!(
                mat.t(),
                Matrix{ elem: vec![
                    vec![11, 21, 31],
                    vec![12, 22, 32]
                ]}
            );
        }

        #[test]
        fn transpose_matrix_1_3() {
            let mat = Matrix{ elem: vec![ vec![1, 2, 3]] };
            assert_eq!(
                mat.t(),
                Matrix{ elem: vec![
                    vec![1],
                    vec![2],
                    vec![3]
                ]}
            );
        }

        #[test]
        fn transpose_matrix_3_1() {
            let mat =
                Matrix{ elem: vec![
                    vec![1],
                    vec![2],
                    vec![3]
                ]};
            assert_eq!(
                mat.t(),
                Matrix{ elem: vec![ vec![1, 2, 3]] }
            );
        }
    }

    #[test]
    fn scalar_mul() {
        let mat = Matrix{ elem: vec![
            vec![1, 2],
            vec![0, -5]
        ]};
        assert_eq!(
            mat.scalar_mul(3),
            Matrix{elem: vec![
                vec![3, 6],
                vec![0, -15]
            ]}
        );
    }

    mod pow {
        use super::*;

        #[test]
        #[should_panic(expected = "only square matrices can be powered")]
        fn non_square_panic() {
            let non_square = Matrix{ elem: vec![
                vec![1, 2]
            ]};
            non_square.pow(3);
        }

        #[test]
        #[should_panic(expected = "pow by 0 not allowed")]
        fn power_by_0_panic() {
            let mat = Matrix{ elem: vec![
                vec![1, 2],
                vec![0, -1]
            ]};
            mat.pow(0);
        }

        #[test]
        fn power_by_1_return_original() {
            let mat = Matrix{ elem: vec![
                vec![0, -1],
                vec![1, 0]
            ]};
            assert_eq!(
                mat.clone().pow(1),
                mat
            );
        }

        #[test]
        fn power_by_even_number() {
            let mat = Matrix{ elem: vec![
                vec![0, -1],
                vec![1, 0]
            ]};
            assert_eq!(
                mat.pow(4),
                Matrix::e(2)
            );
        }

        #[test]
        fn power_by_odd_number() {
            let mat = Matrix{ elem: vec![
                vec![0, -1],
                vec![1, 0]
            ]};
            assert_eq!(
                mat.pow(5),
                mat
            );
        }
    }

    mod move_add {
        use super::*;

        #[test]
        #[should_panic]
        fn different_size_panic() {
            let mat1 = Matrix{ elem: vec![
                vec![11, 12],
                vec![21, 22]
            ]};
            let mat2 = Matrix{ elem: vec![
                vec![11, 12, 13],
                vec![21, 22, 23]
            ]};
            let _ = mat1 + mat2;
        }

        #[test]
        fn return_added() {
            let mat1 = Matrix{ elem: vec![
                vec![1, 2],
                vec![3, 4]
            ]};
            let mat2 = Matrix{ elem: vec![
                vec![10, 20],
                vec![30, 40]
            ]};
            assert_eq!(
                mat1 + mat2,
                Matrix{ elem: vec![
                    vec![11, 22],
                    vec![33, 44]
                ]}
            );
        }
    }

    mod ref_add {
        use super::*;

        #[test]
        #[should_panic]
        fn different_size_panic() {
            let mat1 = Matrix{ elem: vec![
                vec![11, 12],
                vec![21, 22]
            ]};
            let mat2 = Matrix{ elem: vec![
                vec![11, 12, 13],
                vec![21, 22, 23]
            ]};
            let _ = &mat1 + &mat2;
        }

        #[test]
        fn return_added() {
            let mat1 = Matrix{ elem: vec![
                vec![1, 2],
                vec![3, 4]
            ]};
            let mat2 = Matrix{ elem: vec![
                vec![10, 20],
                vec![30, 40]
            ]};
            assert_eq!(
                &mat1 + &mat2,
                Matrix{ elem: vec![
                    vec![11, 22],
                    vec![33, 44]
                ]}
            );
        }
    }
}
