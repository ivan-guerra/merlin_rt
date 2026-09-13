use approx::abs_diff_eq;
use std::{
    fmt::Display,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GeometryError {
    #[error("Cannot normalize a zero vector")]
    ZeroVectorNormalization,

    #[error("Matrix dimensions do not match for multiplication")]
    MatrixDimensionMismatch,

    #[error(
    "Matrix data has {actual} elements, but a {rows}x{cols} matrix \
     requires {}",
    rows * cols
    )]
    MatrixDataLengthMismatch {
        rows: usize,
        cols: usize,
        actual: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3 { x, y, z }
    }
}

impl PartialEq for Point3 {
    fn eq(&self, other: &Self) -> bool {
        abs_diff_eq!(self.x, other.x)
            && abs_diff_eq!(self.y, other.y)
            && abs_diff_eq!(self.z, other.z)
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Point3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub<Point3> for Point3 {
    type Output = Vec3;

    fn sub(self, rhs: Point3) -> Self::Output {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Point3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Display for Point3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write up to 3 decimal places for better readability
        write!(f, "Point({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Result<Vec3, GeometryError> {
        let mag = self.magnitude();
        if mag == 0.0 {
            return Err(GeometryError::ZeroVectorNormalization);
        }
        Ok(Vec3::new(self.x / mag, self.y / mag, self.z / mag))
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        abs_diff_eq!(self.x, other.x)
            && abs_diff_eq!(self.y, other.y)
            && abs_diff_eq!(self.z, other.z)
    }
}

impl Add<Point3> for Vec3 {
    type Output = Point3;

    fn add(self, rhs: Point3) -> Self::Output {
        Point3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write up to 3 decimal places for better readability
        write!(f, "Vector({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, GeometryError> {
        if data.len() != rows * cols {
            return Err(GeometryError::MatrixDataLengthMismatch {
                rows,
                cols,
                actual: data.len(),
            });
        }

        Ok(Matrix { rows, cols, data })
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed[j][i] = self[i][j];
            }
        }
        transposed
    }

    pub fn submatrix(&self, row: usize, col: usize) -> Result<Matrix, GeometryError> {
        if row >= self.rows || col >= self.cols {
            return Err(GeometryError::MatrixDimensionMismatch);
        }

        let mut submatrix = Matrix::new(self.rows - 1, self.cols - 1);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if i != row && j != col {
                    let sub_i = if i < row { i } else { i - 1 };
                    let sub_j = if j < col { j } else { j - 1 };
                    submatrix[sub_i][sub_j] = self[i][j];
                }
            }
        }
        Ok(submatrix)
    }

    pub fn cofactor(&self, row: usize, col: usize) -> Result<f64, GeometryError> {
        let submatrix = self.submatrix(row, col)?;
        let sign = if (row + col).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        Ok(sign * submatrix.determinant()?)
    }

    pub fn determinant(&self) -> Result<f64, GeometryError> {
        if self.rows != self.cols {
            return Err(GeometryError::MatrixDimensionMismatch);
        }

        if self.rows == 2 {
            return Ok(self[0][0] * self[1][1] - self[0][1] * self[1][0]);
        }

        let mut det = 0.0;
        for col in 0..self.cols {
            det += self[0][col] * self.cofactor(0, col)?;
        }
        Ok(det)
    }

    pub fn is_invertible(&self) -> Result<bool, GeometryError> {
        let det = self.determinant()?;
        Ok(!abs_diff_eq!(det, 0.0))
    }

    pub fn inverse(&self) -> Result<Option<Matrix>, GeometryError> {
        if !self.is_invertible()? {
            return Ok(None);
        }

        let mut inverse = Matrix::new(self.rows, self.cols);
        let det = self.determinant()?;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cofactor = self.cofactor(row, col)?;
                inverse[col][row] = cofactor / det;
            }
        }
        Ok(Some(inverse))
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| abs_diff_eq!(a, b))
    }
}

impl Index<usize> for Matrix {
    type Output = [f64];

    fn index(&self, row: usize) -> &Self::Output {
        let start = row * self.cols;
        let end = start + self.cols;
        &self.data[start..end]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let start = row * self.cols;
        let end = start + self.cols;
        &mut self.data[start..end]
    }
}

fn multiply_matrices(lhs: &Matrix, rhs: &Matrix) -> Result<Matrix, GeometryError> {
    if lhs.cols != rhs.rows {
        return Err(GeometryError::MatrixDimensionMismatch);
    }

    let mut result = Matrix::new(lhs.rows, rhs.cols);

    for i in 0..lhs.rows {
        for j in 0..rhs.cols {
            let mut sum = 0.0;
            for k in 0..lhs.cols {
                sum += lhs[i][k] * rhs[k][j];
            }
            result[i][j] = sum;
        }
    }

    Ok(result)
}

impl Mul<Matrix> for Matrix {
    type Output = Result<Matrix, GeometryError>;

    fn mul(self, rhs: Matrix) -> Self::Output {
        multiply_matrices(&self, &rhs)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Result<Matrix, GeometryError>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        multiply_matrices(self, rhs)
    }
}

impl From<Point3> for Matrix {
    fn from(point: Point3) -> Self {
        Matrix {
            rows: 4,
            cols: 1,
            data: vec![point.x, point.y, point.z, 1.0],
        }
    }
}

impl From<Vec3> for Matrix {
    fn from(vector: Vec3) -> Self {
        Matrix {
            rows: 4,
            cols: 1,
            data: vec![vector.x, vector.y, vector.z, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_point_creation() {
        let p = Point3::new(4.3, -4.2, 3.1);
        assert_abs_diff_eq!(p.x, 4.3);
        assert_abs_diff_eq!(p.y, -4.2);
        assert_abs_diff_eq!(p.z, 3.1);
    }

    #[test]
    fn test_point_equality() {
        let p1 = Point3::new(1.0, 2.0, 3.0);
        let p2 = Point3::new(1.0, 2.0, 3.0);
        let p3 = Point3::new(1.0, 2.0, 3.1);
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_point_vector_addition() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(4.0, 5.0, 6.0);
        let result = p + v;
        let result2 = v + p;
        assert_eq!(result, Point3::new(5.0, 7.0, 9.0));
        assert_eq!(result2, Point3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_point_subtraction() {
        let p1 = Point3::new(4.0, 5.0, 6.0);
        let p2 = Point3::new(1.0, 2.0, 3.0);
        let result = p1 - p2;
        assert_eq!(result, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_point_vector_subtraction() {
        let p = Point3::new(4.0, 5.0, 6.0);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = p - v;
        assert_eq!(result, Point3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector_creation() {
        let v = Vec3::new(4.3, -4.2, 3.1);
        assert_abs_diff_eq!(v.x, 4.3);
        assert_abs_diff_eq!(v.y, -4.2);
        assert_abs_diff_eq!(v.z, 3.1);
    }

    #[test]
    fn test_vector_equality() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        let v3 = Vec3::new(1.0, 2.0, 3.1);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let result = v1 + v2;
        assert_eq!(result, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vec3::new(4.0, 5.0, 6.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        let result = v1 - v2;
        assert_eq!(result, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector_negation() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let result = -v;
        assert_eq!(result, Vec3::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let result = v * 3.0;
        assert_eq!(result, Vec3::new(3.0, -6.0, 9.0));
        let result2 = 3.0 * v;
        assert_eq!(result2, Vec3::new(3.0, -6.0, 9.0));
    }

    #[test]
    fn test_vector_scalar_division() {
        let v = Vec3::new(3.0, -6.0, 9.0);
        let result = v / 3.0;
        assert_eq!(result, Vec3::new(1.0, -2.0, 3.0));
    }

    #[test]
    fn test_vector_magnitude() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let v3 = Vec3::new(0.0, 0.0, 1.0);
        let v4 = Vec3::new(1.0, 2.0, 3.0);
        let v5 = Vec3::new(-1.0, -2.0, -3.0);
        assert_abs_diff_eq!(v1.magnitude(), 1.0);
        assert_abs_diff_eq!(v2.magnitude(), 1.0);
        assert_abs_diff_eq!(v3.magnitude(), 1.0);
        assert_abs_diff_eq!(v4.magnitude(), (14.0f64).sqrt());
        assert_abs_diff_eq!(v5.magnitude(), (14.0f64).sqrt());
    }

    #[test]
    fn test_vector_normalization() {
        let v = Vec3::new(4.0, 0.0, 0.0);
        let normalized = v.normalize().unwrap();
        assert_eq!(normalized, Vec3::new(1.0, 0.0, 0.0));

        let v2 = Vec3::new(1.0, 2.0, 3.0);
        let normalized2 = v2.normalize().unwrap();
        let mag = (14.0f64).sqrt();
        assert_eq!(normalized2, Vec3::new(1.0 / mag, 2.0 / mag, 3.0 / mag));

        let zero_vector = Vec3::new(0.0, 0.0, 0.0);
        let result = zero_vector.normalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, -5.0, 6.0);
        let dot_product = v1.dot(&v2);
        assert_abs_diff_eq!(dot_product, 12.0);

        let u1 = Vec3::new(1.0, 0.0, 0.0);
        let u2 = Vec3::new(-1.0, 0.0, 0.0);
        let dot_product2 = u1.dot(&u2);
        assert_abs_diff_eq!(dot_product2, -1.0);
    }

    #[test]
    fn test_vector_cross_product() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let cross_product = v1.cross(&v2);
        assert_eq!(cross_product, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_matrix_creation() {
        let m1 = Matrix::new(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(m1[i][j], 0.0);
            }
        }

        let m2 = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0, 11.0, 12.0, 13.5, 14.5, 15.5,
                16.5,
            ],
        )
        .expect("matrix dimensions should match data length");
        let expected = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.5, 6.5, 7.5, 8.5, 9.0, 10.0, 11.0, 12.0, 13.5, 14.5, 15.5,
                16.5,
            ],
        )
        .expect("matrix dimensions should match data length");
        assert_eq!(m2, expected);
    }

    #[test]
    fn test_matrix_indexing() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .expect("matrix dimensions should match data length");
        assert_abs_diff_eq!(m[0][0], 1.0);
        assert_abs_diff_eq!(m[0][1], 2.0);
        assert_abs_diff_eq!(m[1][0], 3.0);
        assert_abs_diff_eq!(m[1][1], 4.0);
    }

    #[test]
    fn test_matrix_indexing_out_of_bounds() {
        let m = Matrix::new(2, 2);
        let result = std::panic::catch_unwind(|| {
            let _ = m[2][0];
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_equality() {
        let m1 = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        let m2 = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        assert_eq!(m1, m2);

        let m3 = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.1,
            ],
        )
        .expect("matrix dimensions should match data length");
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("matrix dimensions should match data length");
        let m2 = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("matrix dimensions should match data length");
        let result = (m1 * m2).expect("matrix multiplication should succeed");
        let expected = Matrix::from_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0])
            .expect("matrix dimensions should match data length");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiplication_dimension_mismatch() {
        let m1 = Matrix::new(2, 3);
        let m2 = Matrix::new(4, 2);
        let result = m1 * m2;
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_multiplication_with_identity() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .expect("matrix dimensions should match data length");
        let identity = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .expect("matrix dimensions should match data length");
        let result = (&m * &identity).expect("matrix multiplication should succeed");
        assert_eq!(result, m);
    }

    #[test]
    fn test_matrix_from_point() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let m: Matrix = p.into();
        let expected = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 1.0])
            .expect("matrix dimensions should match data length");
        assert_eq!(m.rows, 4);
        assert_eq!(m.cols, 1);
        assert_eq!(m, expected);
    }

    #[test]
    fn test_matrix_from_vector() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let m: Matrix = v.into();
        let expected = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 0.0])
            .expect("matrix dimensions should match data length");
        assert_eq!(m.rows, 4);
        assert_eq!(m.cols, 1);
        assert_eq!(m, expected);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("matrix dimensions should match data length");
        let transposed = m.transpose();
        let expected = Matrix::from_vec(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("matrix dimensions should match data length");
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_identity_matrix_transpose() {
        let identity = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .expect("matrix dimensions should match data length");
        let transposed = identity.transpose();
        assert_eq!(transposed, identity);
    }

    #[test]
    fn test_matrix_submatrix() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 5.0, 0.0, -3.0, 2.0, 7.0, 0.0, 6.0, -3.0])
            .expect("matrix dimensions should match data length");
        let submatrix = m
            .submatrix(0, 2)
            .expect("submatrix extraction should succeed");
        let expected = Matrix::from_vec(2, 2, vec![-3.0, 2.0, 0.0, 6.0])
            .expect("matrix dimensions should match data length");
        assert_eq!(submatrix.rows, 2);
        assert_eq!(submatrix.cols, 2);
        assert_eq!(submatrix, expected);

        let m = Matrix::from_vec(
            4,
            4,
            vec![
                -6.0, 1.0, 1.0, 6.0, -8.0, 5.0, 8.0, 6.0, -1.0, 0.0, 8.0, 2.0, -7.0, 1.0, -1.0, 1.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        let submatrix = m
            .submatrix(2, 1)
            .expect("submatrix extraction should succeed");
        let expected =
            Matrix::from_vec(3, 3, vec![-6.0, 1.0, 6.0, -8.0, 8.0, 6.0, -7.0, -1.0, 1.0])
                .expect("matrix dimensions should match data length");
        assert_eq!(submatrix.rows, 3);
        assert_eq!(submatrix.cols, 3);
        assert_eq!(submatrix, expected);
    }

    #[test]
    fn test_matrix_submatrix_out_of_bounds() {
        let m = Matrix::new(2, 2);
        let result = m.submatrix(2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_cofactor() {
        let m = Matrix::from_vec(3, 3, vec![3.0, 5.0, 0.0, 2.0, -1.0, -7.0, 6.0, -1.0, 5.0])
            .expect("matrix dimensions should match data length");
        let cofactor = m
            .cofactor(0, 0)
            .expect("cofactor calculation should succeed");
        assert_abs_diff_eq!(cofactor, -12.0);
        let cofactor = m
            .cofactor(1, 0)
            .expect("cofactor calculation should succeed");
        assert_abs_diff_eq!(cofactor, -25.0);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 2.0, 6.0, -5.0, 8.0, -4.0, 2.0, 6.0, 4.0])
            .expect("matrix dimensions should match data length");
        let det = m
            .determinant()
            .expect("determinant calculation should succeed");
        assert_abs_diff_eq!(det, -196.0);

        let m = Matrix::from_vec(
            4,
            4,
            vec![
                -2.0, -8.0, 3.0, 5.0, -3.0, 1.0, 7.0, 3.0, 1.0, 2.0, -9.0, 6.0, -6.0, 7.0, 7.0,
                -9.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        let det = m
            .determinant()
            .expect("determinant calculation should succeed");
        assert_abs_diff_eq!(det, -4071.0);
    }

    #[test]
    fn test_matrix_determinant_non_square() {
        let m = Matrix::new(2, 3);
        let result = m.determinant();
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_invertibility() {
        let m = Matrix::from_vec(
            4,
            4,
            vec![
                6.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 6.0, 4.0, -9.0, 3.0, -7.0, 9.0, 1.0, 7.0, -6.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        let is_invertible = m
            .is_invertible()
            .expect("invertibility check should succeed");
        assert!(is_invertible);

        let m = Matrix::from_vec(
            4,
            4,
            vec![
                -4.0, 2.0, -2.0, -3.0, 9.0, 6.0, 2.0, 6.0, 0.0, -5.0, 1.0, -5.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        let is_invertible = m.is_invertible().expect(
            "invertibility check should
    succeed",
        );
        assert!(!is_invertible);
    }

    #[test]
    fn test_matrix_inverse() {
        let m1 = Matrix::from_vec(
            4,
            4,
            vec![
                8.0, -5.0, 9.0, 2.0, 7.0, 5.0, 6.0, 1.0, -6.0, 0.0, 9.0, 6.0, -3.0, 0.0, -9.0, -4.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        let inverse1 = m1
            .inverse()
            .expect("inverse calculation should succeed")
            .expect("matrix should be invertible");
        let expected = Matrix::from_vec(
            4,
            4,
            vec![
                -0.15384615384615385,
                -0.15384615384615385,
                -0.28205128205128205,
                -0.5384615384615384,
                -0.07692307692307693,
                0.12307692307692308,
                0.02564102564102564,
                0.03076923076923077,
                0.358974358974359,
                0.358974358974359,
                0.4358974358974359,
                0.9230769230769231,
                -0.6923076923076923,
                -0.6923076923076923,
                -0.7692307692307693,
                -1.9230769230769231,
            ],
        )
        .expect("matrix dimensions should match data length");
        assert_eq!(inverse1, expected);

        let m2 = Matrix::from_vec(
            4,
            4,
            vec![
                9.0, 3.0, 0.0, 9.0, -5.0, -2.0, -6.0, -3.0, -4.0, 9.0, 6.0, 4.0, -7.0, 6.0, 6.0,
                2.0,
            ],
        )
        .expect("matrix dimensions should match data length");
        let inverse2 = m2
            .inverse()
            .expect("inverse calculation should succeed")
            .expect("matrix should be invertible");
        let expected = Matrix::from_vec(
            4,
            4,
            vec![
                -0.040740740740740744,
                -0.07777777777777778,
                0.14444444444444443,
                -0.2222222222222222,
                -0.07777777777777778,
                0.03333333333333333,
                0.36666666666666664,
                -0.3333333333333333,
                -0.029012345679012345,
                -0.1462962962962963,
                -0.10925925925925926,
                0.12962962962962962,
                0.17777777777777778,
                0.06666666666666667,
                -0.26666666666666666,
                0.3333333333333333,
            ],
        )
        .expect("matrix dimensions should match data length");
        assert_eq!(inverse2, expected);
    }
}
