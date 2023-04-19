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

    pub fn det(&self) -> T {
        assert!(
            self.is_square(),
            "det is only defined for square matrices"
        );

        if self.shape().0 == 1 {
            return self.elem[0][0];
        } else {
            self.elem[0].iter().enumerate().map(|(j, &a1j)| {
                let j = j+1;
                if (1+j) % 2 == 0 {
                    a1j * self.submat(1,j).det()
                } else {
                    - a1j * self.submat(1,j).det()
                }
            }).sum()
        }
    }

    fn submat(&self, i: usize, j: usize) -> Self {
        let (m, n) = self.shape();
        assert!(m > 1 && n > 1);

        let i = i-1;
        let j = j-1;

        let mut v = Vec::new();
        for s in 0..m {
            if s == i {
                continue;
            }
            v.push(Vec::new());
            for t in 0..n {
                if t == j {
                    continue;
                }
                v.last_mut().unwrap().push(self.elem[s][t]);
            }
        }
        Matrix::from_vec(v)
    }

    fn cofactor(&self, i: usize, j: usize) -> T {
        let (m,n) = self.shape();

        assert!(
            self.is_square(),
            "cofactor is only defined for square metrices"
        );
        assert!(
            m > 1,
            "cofactor is not defined for (1,1) matrices"
        );
        assert!(
            i <= m && j <= n,
            "index out of range"
        );

        if (i+j) % 2 == 0 {
            self.submat(i, j).det()
        } else {
            - self.submat(i, j).det()
        }
    }

    pub fn cofactor_mat(&self) -> Self {
        let mut v = Vec::new();
        let (m, n) = self.shape();

        assert!(
            self.is_square(),
            "cofactor matrix is only defined for square metrices"
        );
        assert!(
            m > 1,
            "cofactor matrix is not defined for (1,1) matrices"
        );

        for i in 1..=n {
            v.push(Vec::new());
            for j in 1..=m {
                v.last_mut().unwrap().push(self.cofactor(j, i));
            }
        }
        Matrix::from_vec(v)
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
        if let Some(m) = zip_with(|x, y| x + y, self, rhs) {
            m
        } else {
            panic!("you can only add matrices with the same shape");
        }
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
        if let Some(m) = zip_with(|x, y| x - y, self, rhs) {
            m
        } else {
            panic!("you can only subtract matrices with the same shape");
        }
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

        assert_eq!(
            m, m2,
            "lhs width must match rhs height"
        );

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

fn zip_with<F, T: Num, U: Num>(f: F, m1: &Matrix<T>, m2: &Matrix<T>) -> Option<Matrix<U>>
where F: Fn(T, T) -> U {
    if m1.shape() != m2.shape() {
        return None;
    }

    let v =
    m1.elem.iter().zip(m2.elem.iter()).map(
        |(v1, v2)| v1.iter().zip(v2.iter()).map(
            |(e1, e2)| f(*e1, *e2)
        ).collect()
    ).collect();
    Some(Matrix::from_vec(v))
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
            let matrix_2_3 = Matrix::from_vec(vec![
                vec![1, 2, 3],
                vec![1, 3, 2],
            ]);
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
            let mat_2_2 = Matrix::from_vec(vec![
                vec![1, 2],
                vec![4, -2]
            ]);
            assert!(mat_2_2.is_square());
        }

        #[test]
        fn matrix_1_2_is_not_square() {
            let mat_1_2 = Matrix::from_vec(vec![
                vec![1, 2]
            ]);
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
                Matrix::from_vec(vec![ vec![0; 3]; 2])
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
                Matrix::from_vec(vec![vec![1]])
            );
        }

        #[test]
        fn n3_create_e3() {
            let mat: Matrix<i32> = Matrix::e(3);
            assert_eq!(
                mat,
                Matrix::from_vec(vec![
                    vec![1, 0, 0],
                    vec![0, 1, 0],
                    vec![0, 0, 1]
                ])
            );
        }
    }

    mod t {
        use super::*;

        #[test]
        fn transpose_matrix_2_3() {
            let mat = Matrix::from_vec(vec![
                vec![11, 12],
                vec![21, 22],
                vec![31, 32]
            ]);
            assert_eq!(
                mat.t(),
                Matrix::from_vec(vec![
                    vec![11, 21, 31],
                    vec![12, 22, 32]
                ])
            );
        }

        #[test]
        fn transpose_matrix_1_3() {
            let mat = Matrix::from_vec(vec![ vec![1, 2, 3]]);
            assert_eq!(
                mat.t(),
                Matrix::from_vec(vec![
                    vec![1],
                    vec![2],
                    vec![3]
                ])
            );
        }

        #[test]
        fn transpose_matrix_3_1() {
            let mat =
                Matrix::from_vec(vec![
                    vec![1],
                    vec![2],
                    vec![3]
                ]);
            assert_eq!(
                mat.t(),
                Matrix::from_vec(vec![ vec![1, 2, 3]])
            );
        }
    }

    #[test]
    fn scalar_mul() {
        let mat = Matrix::from_vec(vec![
            vec![1, 2],
            vec![0, -5]
        ]);
        assert_eq!(
            mat.scalar_mul(3),
            Matrix::from_vec(vec![
                vec![3, 6],
                vec![0, -15]
            ])
        );
    }

    mod pow {
        use super::*;

        #[test]
        #[should_panic(expected = "only square matrices can be powered")]
        fn non_square_matrix_panic() {
            let non_square_mat = Matrix::from_vec(vec![
                vec![1, 2]
            ]);
            non_square_mat.pow(3);
        }

        #[test]
        #[should_panic(expected = "pow by 0 not allowed")]
        fn power_by_0_panic() {
            let mat = Matrix::from_vec(vec![
                vec![1, 2],
                vec![0, -1]
            ]);
            mat.pow(0);
        }

        #[test]
        fn power_by_1_return_original() {
            let mat = Matrix::from_vec(vec![
                vec![0, -1],
                vec![1, 0]
            ]);
            assert_eq!(
                mat.clone().pow(1),
                mat
            );
        }

        #[test]
        fn power_by_even_number() {
            let mat = Matrix::from_vec(vec![
                vec![0, -1],
                vec![1, 0]
            ]);
            assert_eq!(
                mat.pow(4),
                Matrix::e(2)
            );
        }

        #[test]
        fn power_by_odd_number() {
            let mat = Matrix::from_vec(vec![
                vec![0, -1],
                vec![1, 0]
            ]);
            assert_eq!(
                mat.pow(5),
                mat
            );
        }
    }

    mod move_add {
        use super::*;

        #[test]
        #[should_panic(expected = "you can only add matrices with the same shape")]
        fn different_size_panic() {
            let mat1 = Matrix::from_vec(vec![
                vec![11, 12],
                vec![21, 22]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![11, 12, 13],
                vec![21, 22, 23]
            ]);
            let _ = mat1 + mat2;
        }

        #[test]
        fn same_size_return_added() {
            let mat1 = Matrix::from_vec(vec![
                vec![1, 2],
                vec![3, 4]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![10, 20],
                vec![30, 40]
            ]);
            assert_eq!(
                mat1 + mat2,
                Matrix::from_vec(vec![
                    vec![11, 22],
                    vec![33, 44]
                ])
            );
        }
    }

    mod ref_add {
        use super::*;

        #[test]
        #[should_panic(expected = "you can only add matrices with the same shape")]
        fn different_size_panic() {
            let mat1 = Matrix::from_vec(vec![
                vec![11, 12],
                vec![21, 22]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![11, 12, 13],
                vec![21, 22, 23]
            ]);
            let _ = &mat1 + &mat2;
        }

        #[test]
        fn same_size_return_added() {
            let mat1 = Matrix::from_vec(vec![
                vec![1, 2],
                vec![3, 4]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![10, 20],
                vec![30, 40]
            ]);
            assert_eq!(
                &mat1 + &mat2,
                Matrix::from_vec(vec![
                    vec![11, 22],
                    vec![33, 44]
                ])
            );
        }
    }

    mod move_sub {
        use super::*;

        #[test]
        #[should_panic(expected = "you can only subtract matrices with the same shape")]
        fn different_size_panic() {
            let mat1 = Matrix::from_vec(vec![
                vec![11, 12],
                vec![21, 22]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![11, 12, 13],
                vec![21, 22, 23]
            ]);
            let _ = mat1 - mat2;
        }

        #[test]
        fn same_size_return_subtracted() {
            let mat1 = Matrix::from_vec(vec![
                vec![1, 2],
                vec![3, 4]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![10, 20],
                vec![30, 40]
            ]);
            assert_eq!(
                mat1 - mat2,
                Matrix::from_vec(vec![
                    vec![-9, -18],
                    vec![-27, -36]
                ])
            );
        }
    }

    mod ref_sub {
        use super::*;

        #[test]
        #[should_panic(expected = "you can only subtract matrices with the same shape")]
        fn different_size_panic() {
            let mat1 = Matrix::from_vec(vec![
                vec![11, 12],
                vec![21, 22]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![11, 12, 13],
                vec![21, 22, 23]
            ]);
            let _ = &mat1 - &mat2;
        }

        #[test]
        fn same_size_return_subtracted() {
            let mat1 = Matrix::from_vec(vec![
                vec![1, 2],
                vec![3, 4]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![10, 20],
                vec![30, 40]
            ]);
            assert_eq!(
                &mat1 - &mat2,
                Matrix::from_vec(vec![
                    vec![-9, -18],
                    vec![-27, -36]
                ])
            );
        }
    }

    mod move_mul {
        use super::*;

        #[test]
        #[should_panic(expected = "lhs width must match rhs height")]
        fn lhs_width_rhs_height_unmatch_panic() {
            let mat1 = Matrix::from_vec(vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
                vec![7, 8, 9]
            ]);
            let _ = mat1 * mat2;
        }

        #[test]
        fn match_return_product() {
            let mat1 = Matrix::from_vec(vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6]
            ]);
            let mat2 = Matrix::from_vec(vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
            ]);
            let prod = mat1 * mat2;
            assert_eq!(
                prod,
                Matrix::from_vec(vec![
                    vec![9, 12, 15],
                    vec![19, 26, 33],
                    vec![29, 40, 51]
                ])
            )
        }
    }

    #[test]
    fn c_return_conjugate() {
        let m = Matrix::from_vec(vec![
            vec![Complex::new(1,2), Complex::new(2,-3)],
            vec![Complex::new(1,0), Complex::new(0,3)],
            vec![Complex::new(0,-5), Complex::new(2,1)],
        ]);
        assert_eq!(
            m.c(),
            Matrix::from_vec(vec![
                vec![Complex::new(1,-2), Complex::new(2,3)],
                vec![Complex::new(1,0), Complex::new(0,-3)],
                vec![Complex::new(0,5), Complex::new(2,-1)],
            ])
        )
    }

    #[test]
    fn hermite() {
        let m = Matrix::from_vec(vec![
            vec![Complex::new(1,2), Complex::new(2,-3)],
            vec![Complex::new(1,0), Complex::new(0,3)],
            vec![Complex::new(0,-5), Complex::new(2,1)],
        ]);
        assert_eq!(
            m.hermite(),
            Matrix::from_vec(vec![
                vec![Complex::new(1,-2), Complex::new(1,0), Complex::new(0,5)],
                vec![Complex::new(2,3), Complex::new(0,-3), Complex::new(2,-1)]
            ])
        )
    }

    mod det {
        use super::*;

        #[test]
        #[should_panic(expected = "det is only defined for square matrices")]
        fn non_square_panic() {
            let non_square = Matrix::from_vec(vec![
                vec![1, 2, 3],
                vec![4, 5, 6]
            ]);
            non_square.det();
        }

        #[test]
        fn one() {
            let m = Matrix::from_vec(vec![vec![2]]);
            assert_eq!(
                m.det(),
                2
            );
        }

        #[test]
        fn two() {
            let m = Matrix::from_vec(vec![
                vec![1, 2],
                vec![3, 4]
            ]);
            assert_eq!(
                m.det(),
                1 * 4 - 2 * 3
            )
        }

        #[test]
        fn three() {
            let m = Matrix::from_vec(vec![
                vec![2, 1, 0],
                vec![3, 1, 2],
                vec![-1, 0, 5]
            ]);
            assert_eq!(
                m.det(),
                -7
            )
        }
    }

    mod cofactor {
        use super::*;

        #[test]
        #[should_panic(expected = "cofactor is only defined for square metrices")]
        fn non_square_panic() {
            let m = Matrix::from_vec(vec![
                vec![1,2,3],
                vec![4,5,6]
            ]);
            m.cofactor(1,2);
        }

        #[test]
        #[should_panic(expected = "cofactor is not defined for (1,1) matrices")]
        fn one_one_matrix_panic() {
            let m = Matrix::from_vec(vec![vec![2]]);
            m.cofactor(1,1);
        }

        #[test]
        #[should_panic(expected = "index out of range")]
        fn index_out_of_range_panic() {
            let m = Matrix::from_vec(vec![
                vec![1,2,3],
                vec![4,5,6],
                vec![7,8,9]
            ]);
            m.cofactor(4,1);
        }

        #[test]
        fn calc() {
            let m = Matrix::from_vec(vec![
                vec![1,2,3],
                vec![4,5,6],
                vec![7,8,9]
            ]);
            assert_eq!(
                m.cofactor(1,1),
                -3
            );
            assert_eq!(
                m.cofactor(2,3),
                6
            );
        }
    }

    mod cofactor_mat {
        use super::*;

        #[test]
        #[should_panic(expected = "cofactor matrix is only defined for square metrices")]
        fn non_square_panic() {
            let m = Matrix::from_vec(vec![
                vec![1,2,3],
                vec![4,5,6]
            ]);
            m.cofactor_mat();
        }

        #[test]
        #[should_panic(expected = "cofactor matrix is not defined for (1,1) matrices")]
        fn one_one_matrix_panic() {
            let m = Matrix::from_vec(vec![vec![2]]);
            m.cofactor_mat();
        }

        #[test]
        fn calc() {
            let m = Matrix::from_vec(vec![
                vec![1,2,3],
                vec![4,5,6],
                vec![7,8,9]
            ]);
            assert_eq!(
                m.cofactor_mat(),
                Matrix::from_vec(vec![
                    vec![-3, 6, -3],
                    vec![6, -12, 6],
                    vec![-3, 6, -3]
                ])
            );
        }
    }
}
