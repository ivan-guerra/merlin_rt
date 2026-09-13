use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy)]
struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    fn new(r: f64, g: f64, b: f64) -> Self {
        Color { r, g, b }
    }
}

impl Add<Color> for Color {
    type Output = Color;

    fn add(self, rhs: Color) -> Self::Output {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl Sub<Color> for Color {
    type Output = Color;

    fn sub(self, rhs: Color) -> Self::Output {
        Color {
            r: self.r - rhs.r,
            g: self.g - rhs.g,
            b: self.b - rhs.b,
        }
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, rhs: f64) -> Self::Output {
        Color {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}

impl Mul<Color> for f64 {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color {
            r: self * rhs.r,
            g: self * rhs.g,
            b: self * rhs.b,
        }
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_color_new() {
        let color = Color::new(-0.5, 0.25, 0.75);
        assert_abs_diff_eq!(color.r, -0.5);
        assert_abs_diff_eq!(color.g, 0.25);
        assert_abs_diff_eq!(color.b, 0.75);
    }

    #[test]
    fn test_color_add() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);
        let result = c1 + c2;
        assert_abs_diff_eq!(result.r, 1.6);
        assert_abs_diff_eq!(result.g, 0.7);
        assert_abs_diff_eq!(result.b, 1.0);
    }

    #[test]
    fn test_color_sub() {
        let c1 = Color::new(1.0, 2.0, 3.0);
        let c2 = Color::new(0.5, 1.0, 1.5);
        let result = c1 - c2;
        assert_abs_diff_eq!(result.r, 0.5);
        assert_abs_diff_eq!(result.g, 1.0);
        assert_abs_diff_eq!(result.b, 1.5);
    }

    #[test]
    fn test_color_mul_scalar() {
        let c = Color::new(0.2, 0.3, 0.4);
        let result = c * 2.0;
        assert_abs_diff_eq!(result.r, 0.4);
        assert_abs_diff_eq!(result.g, 0.6);
        assert_abs_diff_eq!(result.b, 0.8);
        let result2 = 2.0 * c;
        assert_abs_diff_eq!(result2.r, 0.4);
        assert_abs_diff_eq!(result2.g, 0.6);
        assert_abs_diff_eq!(result2.b, 0.8);
    }

    #[test]
    fn test_color_mul_color() {
        let c1 = Color::new(1.0, 0.2, 0.4);
        let c2 = Color::new(0.9, 1.0, 0.1);
        let result = c1 * c2;
        assert_abs_diff_eq!(result.r, 0.9);
        assert_abs_diff_eq!(result.g, 0.2);
        assert_abs_diff_eq!(result.b, 0.04);
    }
}
