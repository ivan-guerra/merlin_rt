use std::fs::File;
use std::io::Write;
use std::ops::{Add, Mul, Sub};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CanvasError {
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    #[error("Failed to write PPM data: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
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

#[derive(Debug)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Color>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        let pixels = vec![Color::new(0.0, 0.0, 0.0); width * height];
        Canvas {
            width,
            height,
            pixels,
        }
    }

    pub fn write_pixel(&mut self, x: usize, y: usize, color: Color) -> Result<(), CanvasError> {
        if x >= self.width || y >= self.height {
            return Err(CanvasError::IndexOutOfBounds(format!(
                "Coordinates ({}, {}) are out of bounds for canvas of size {}x{}",
                x, y, self.width, self.height
            )));
        }
        let index = y * self.width + x;
        self.pixels[index] = color;
        Ok(())
    }

    pub fn write_to_ppm(&self, filename: &Path) -> Result<(), CanvasError> {
        let mut file = File::create(filename)?;
        let header = self.construct_ppm_header();
        let pixel_data = self.construct_ppm_pixel_data();
        file.write_all(header.as_bytes())?;
        file.write_all(pixel_data.as_bytes())?;
        Ok(())
    }

    fn construct_ppm_header(&self) -> String {
        format!("P3\n{} {}\n255\n", self.width, self.height)
    }

    fn construct_ppm_pixel_data(&self) -> String {
        const PPM_MAX_LINE_LENGTH: usize = 70;
        let mut ppm_data = String::new();
        let mut line_length = 0;

        for (i, pixel) in self.pixels.iter().enumerate() {
            let components = [
                (pixel.r * 255.0).round().clamp(0.0, 255.0) as u8,
                (pixel.g * 255.0).round().clamp(0.0, 255.0) as u8,
                (pixel.b * 255.0).round().clamp(0.0, 255.0) as u8,
            ];

            for component in components {
                let component = component.to_string();
                let separator_length = usize::from(line_length > 0);

                if line_length + separator_length + component.len() > PPM_MAX_LINE_LENGTH {
                    ppm_data.push('\n');
                    line_length = 0;
                }

                if line_length > 0 {
                    ppm_data.push(' ');
                    line_length += 1;
                }

                ppm_data.push_str(&component);
                line_length += component.len();
            }

            if (i + 1) % self.width == 0 {
                ppm_data.push('\n');
                line_length = 0;
            }
        }

        ppm_data
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

    #[test]
    fn test_canvas_new() {
        let canvas = Canvas::new(10, 20);
        assert_eq!(canvas.width, 10);
        assert_eq!(canvas.height, 20);
        for pixel in canvas.pixels {
            assert_abs_diff_eq!(pixel.r, 0.0);
            assert_abs_diff_eq!(pixel.g, 0.0);
            assert_abs_diff_eq!(pixel.b, 0.0);
        }
    }

    #[test]
    fn test_write_pixel() {
        let mut canvas = Canvas::new(10, 20);
        let red = Color::new(1.0, 0.0, 0.0);
        canvas.write_pixel(2, 3, red).unwrap();
        let index = 3 * canvas.width + 2;
        assert_abs_diff_eq!(canvas.pixels[index].r, 1.0);
        assert_abs_diff_eq!(canvas.pixels[index].g, 0.0);
        assert_abs_diff_eq!(canvas.pixels[index].b, 0.0);

        // Test out of bounds
        let result = canvas.write_pixel(10, 3, red);
        assert!(result.is_err());
    }

    #[test]
    fn test_construct_ppm_header() {
        let canvas = Canvas::new(5, 3);
        let header = canvas.construct_ppm_header();
        let expected_header = "P3\n5 3\n255\n";
        assert_eq!(header, expected_header);
    }

    #[test]
    fn test_construct_ppm_pixel_data() {
        let mut canvas = Canvas::new(5, 3);
        let c1 = Color::new(1.5, 0.0, 0.0);
        let c2 = Color::new(0.0, 0.5, 0.0);
        let c3 = Color::new(-0.5, 0.0, 1.0);
        canvas.write_pixel(0, 0, c1).unwrap();
        canvas.write_pixel(2, 1, c2).unwrap();
        canvas.write_pixel(4, 2, c3).unwrap();
        let pixel_data = canvas.construct_ppm_pixel_data();
        let expected_pixel_data = "\
                                   255 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
                                   0 0 0 0 0 0 0 128 0 0 0 0 0 0 0\n\
                                   0 0 0 0 0 0 0 0 0 0 0 0 0 0 255\n";
        assert_eq!(pixel_data, expected_pixel_data);
    }

    #[test]
    fn test_construct_ppm_pixel_data_line_length() {
        let mut canvas = Canvas::new(10, 2);
        let color = Color::new(1.0, 0.8, 0.6);
        for y in 0..2 {
            for x in 0..10 {
                canvas.write_pixel(x, y, color).unwrap();
            }
        }
        let pixel_data = canvas.construct_ppm_pixel_data();
        let expected_pixel_data = "\
                                   255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n\
                                   153 255 204 153 255 204 153 255 204 153 255 204 153\n\
                                   255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n\
                                   153 255 204 153 255 204 153 255 204 153 255 204 153\n";
        assert_eq!(pixel_data, expected_pixel_data);
    }

    #[test]
    fn test_construct_ppm_pixel_data_terminating_newline() {
        let mut canvas = Canvas::new(5, 3);
        let color = Color::new(1.0, 0.8, 0.6);
        for y in 0..3 {
            for x in 0..5 {
                canvas.write_pixel(x, y, color).unwrap();
            }
        }
        let pixel_data = canvas.construct_ppm_pixel_data();
        assert!(pixel_data.ends_with('\n'));
    }
}
