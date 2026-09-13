use anyhow::{Result, ensure};
use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};

fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < f64::EPSILON
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point { x, y, z }
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        float_eq(self.x, other.x) && float_eq(self.y, other.y) && float_eq(self.z, other.z)
    }
}

impl Add<Vector> for Point {
    type Output = Point;

    fn add(self, rhs: Vector) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub<Point> for Point {
    type Output = Vector;

    fn sub(self, rhs: Point) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Sub<Vector> for Point {
    type Output = Point;

    fn sub(self, rhs: Vector) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write up to 3 decimal places for better readability
        write!(f, "Point({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector { x, y, z }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Result<Self> {
        let mag = self.magnitude();
        ensure!(mag != 0.0, "Cannot normalize a zero vector");
        Ok(Vector::new(self.x / mag, self.y / mag, self.z / mag))
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Vector {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        float_eq(self.x, other.x) && float_eq(self.y, other.y) && float_eq(self.z, other.z)
    }
}

impl Add<Point> for Vector {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Add<Vector> for Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        Vector {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub<Vector> for Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vector {
    type Output = Vector;

    fn neg(self) -> Self::Output {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        Vector {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        Vector {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
        }
    }
}

impl Div<f64> for Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        Vector {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write up to 3 decimal places for better readability
        write!(f, "Vector({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::new(4.3, -4.2, 3.1);
        assert_eq!(p.x, 4.3);
        assert_eq!(p.y, -4.2);
        assert_eq!(p.z, 3.1);
    }

    #[test]
    fn test_point_equality() {
        let p1 = Point::new(1.0, 2.0, 3.0);
        let p2 = Point::new(1.0, 2.0, 3.0);
        let p3 = Point::new(1.0, 2.0, 3.1);
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_point_vector_addition() {
        let p = Point::new(1.0, 2.0, 3.0);
        let v = Vector::new(4.0, 5.0, 6.0);
        let result = p + v;
        let result2 = v + p;
        assert_eq!(result, Point::new(5.0, 7.0, 9.0));
        assert_eq!(result2, Point::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_point_subtraction() {
        let p1 = Point::new(4.0, 5.0, 6.0);
        let p2 = Point::new(1.0, 2.0, 3.0);
        let result = p1 - p2;
        assert_eq!(result, Vector::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_point_vector_subtraction() {
        let p = Point::new(4.0, 5.0, 6.0);
        let v = Vector::new(1.0, 2.0, 3.0);
        let result = p - v;
        assert_eq!(result, Point::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector_creation() {
        let v = Vector::new(4.3, -4.2, 3.1);
        assert_eq!(v.x, 4.3);
        assert_eq!(v.y, -4.2);
        assert_eq!(v.z, 3.1);
    }

    #[test]
    fn test_vector_equality() {
        let v1 = Vector::new(1.0, 2.0, 3.0);
        let v2 = Vector::new(1.0, 2.0, 3.0);
        let v3 = Vector::new(1.0, 2.0, 3.1);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new(1.0, 2.0, 3.0);
        let v2 = Vector::new(4.0, 5.0, 6.0);
        let result = v1 + v2;
        assert_eq!(result, Vector::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector::new(4.0, 5.0, 6.0);
        let v2 = Vector::new(1.0, 2.0, 3.0);
        let result = v1 - v2;
        assert_eq!(result, Vector::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector_negation() {
        let v = Vector::new(1.0, -2.0, 3.0);
        let result = -v;
        assert_eq!(result, Vector::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let v = Vector::new(1.0, -2.0, 3.0);
        let result = v * 3.0;
        assert_eq!(result, Vector::new(3.0, -6.0, 9.0));
        let result2 = 3.0 * v;
        assert_eq!(result2, Vector::new(3.0, -6.0, 9.0));
    }

    #[test]
    fn test_vector_scalar_division() {
        let v = Vector::new(3.0, -6.0, 9.0);
        let result = v / 3.0;
        assert_eq!(result, Vector::new(1.0, -2.0, 3.0));
    }

    #[test]
    fn test_vector_magnitude() {
        let v1 = Vector::new(1.0, 0.0, 0.0);
        let v2 = Vector::new(0.0, 1.0, 0.0);
        let v3 = Vector::new(0.0, 0.0, 1.0);
        let v4 = Vector::new(1.0, 2.0, 3.0);
        let v5 = Vector::new(-1.0, -2.0, -3.0);
        assert_eq!(v1.magnitude(), 1.0);
        assert_eq!(v2.magnitude(), 1.0);
        assert_eq!(v3.magnitude(), 1.0);
        assert_eq!(v4.magnitude(), (14.0f64).sqrt());
        assert_eq!(v5.magnitude(), (14.0f64).sqrt());
    }

    #[test]
    fn test_vector_normalization() {
        let v = Vector::new(4.0, 0.0, 0.0);
        let normalized = v.normalize().unwrap();
        assert_eq!(normalized, Vector::new(1.0, 0.0, 0.0));

        let v2 = Vector::new(1.0, 2.0, 3.0);
        let normalized2 = v2.normalize().unwrap();
        let mag = (14.0f64).sqrt();
        assert_eq!(normalized2, Vector::new(1.0 / mag, 2.0 / mag, 3.0 / mag));

        let zero_vector = Vector::new(0.0, 0.0, 0.0);
        let result = zero_vector.normalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector::new(1.0, 2.0, 3.0);
        let v2 = Vector::new(4.0, -5.0, 6.0);
        let dot_product = v1.dot(&v2);
        assert_eq!(dot_product, 12.0);

        let u1 = Vector::new(1.0, 0.0, 0.0);
        let u2 = Vector::new(-1.0, 0.0, 0.0);
        let dot_product2 = u1.dot(&u2);
        assert_eq!(dot_product2, -1.0);
    }

    #[test]
    fn test_vector_cross_product() {
        let v1 = Vector::new(1.0, 2.0, 3.0);
        let v2 = Vector::new(4.0, 5.0, 6.0);
        let cross_product = v1.cross(&v2);
        assert_eq!(cross_product, Vector::new(-3.0, 6.0, -3.0));
    }
}
