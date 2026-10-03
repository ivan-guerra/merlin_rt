use approx::abs_diff_eq;
use nalgebra::Vector3;
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
pub struct Color(Vector3<f64>);

impl Color {
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Color(Vector3::new(r, g, b))
    }

    pub fn r(&self) -> f64 {
        self.0.x
    }

    pub fn g(&self) -> f64 {
        self.0.y
    }

    pub fn b(&self) -> f64 {
        self.0.z
    }
}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        abs_diff_eq!(self.r(), other.r(), epsilon = 1e-5)
            && abs_diff_eq!(self.g(), other.g(), epsilon = 1e-5)
            && abs_diff_eq!(self.b(), other.b(), epsilon = 1e-5)
    }
}

impl Add<Color> for Color {
    type Output = Color;

    fn add(self, rhs: Color) -> Self::Output {
        Color(Vector3::new(
            self.r() + rhs.r(),
            self.g() + rhs.g(),
            self.b() + rhs.b(),
        ))
    }
}

impl Sub<Color> for Color {
    type Output = Color;

    fn sub(self, rhs: Color) -> Self::Output {
        Color(Vector3::new(
            self.r() - rhs.r(),
            self.g() - rhs.g(),
            self.b() - rhs.b(),
        ))
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color(Vector3::new(
            self.r() * rhs.r(),
            self.g() * rhs.g(),
            self.b() * rhs.b(),
        ))
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, rhs: f64) -> Self::Output {
        Color(Vector3::new(self.r() * rhs, self.g() * rhs, self.b() * rhs))
    }
}

impl Mul<Color> for f64 {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color(Vector3::new(self * rhs.r(), self * rhs.g(), self * rhs.b()))
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
                (pixel.r() * 255.0).round().clamp(0.0, 255.0) as u8,
                (pixel.g() * 255.0).round().clamp(0.0, 255.0) as u8,
                (pixel.b() * 255.0).round().clamp(0.0, 255.0) as u8,
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
    fn test_colors_are_rgb_tuples() {
        let c = Color::new(-0.5, 0.4, 1.7);
        assert_abs_diff_eq!(c.r(), -0.5);
        assert_abs_diff_eq!(c.g(), 0.4);
        assert_abs_diff_eq!(c.b(), 1.7);
    }

    #[test]
    fn test_adding_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);
        let result = Color::new(c1.r() + c2.r(), c1.g() + c2.g(), c1.b() + c2.b());
        assert_abs_diff_eq!(result.r(), 1.6);
        assert_abs_diff_eq!(result.g(), 0.7);
        assert_abs_diff_eq!(result.b(), 1.0);
    }

    #[test]
    fn test_subtracting_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);
        let result = Color::new(c1.r() - c2.r(), c1.g() - c2.g(), c1.b() - c2.b());
        assert_abs_diff_eq!(result.r(), 0.2);
        assert_abs_diff_eq!(result.g(), 0.5);
        assert_abs_diff_eq!(result.b(), 0.5);
    }

    #[test]
    fn test_multiplying_color_by_scalar() {
        let c = Color::new(0.2, 0.3, 0.4);
        let scalar = 2.0;
        let result = Color::new(c.r() * scalar, c.g() * scalar, c.b() * scalar);
        assert_abs_diff_eq!(result.r(), 0.4);
        assert_abs_diff_eq!(result.g(), 0.6);
        assert_abs_diff_eq!(result.b(), 0.8);
    }

    #[test]
    fn test_multiplying_colors() {
        let c1 = Color::new(1.0, 0.2, 0.4);
        let c2 = Color::new(0.9, 1.0, 0.1);
        let result = c1 * c2;
        assert_abs_diff_eq!(result.r(), 0.9);
        assert_abs_diff_eq!(result.g(), 0.2);
        assert_abs_diff_eq!(result.b(), 0.04);
    }

    #[test]
    fn test_creating_a_canvas() {
        let canvas = Canvas::new(10, 20);
        assert_eq!(canvas.width, 10);
        assert_eq!(canvas.height, 20);
        for pixel in canvas.pixels {
            assert_abs_diff_eq!(pixel.r(), 0.0);
            assert_abs_diff_eq!(pixel.g(), 0.0);
            assert_abs_diff_eq!(pixel.b(), 0.0);
        }
    }

    #[test]
    fn test_writing_pixels_to_a_canvas() {
        let mut canvas = Canvas::new(10, 20);
        let red = Color::new(1.0, 0.0, 0.0);
        canvas.write_pixel(2, 3, red).unwrap();
        let index = 3 * canvas.width + 2;
        assert_abs_diff_eq!(canvas.pixels[index].r(), 1.0);
        assert_abs_diff_eq!(canvas.pixels[index].g(), 0.0);
        assert_abs_diff_eq!(canvas.pixels[index].b(), 0.0);

        // Test out of bounds
        let result = canvas.write_pixel(10, 3, red);
        assert!(result.is_err());
    }

    #[test]
    fn test_constructing_the_ppm_header() {
        let canvas = Canvas::new(5, 3);
        let header = canvas.construct_ppm_header();
        let expected_header = "P3\n5 3\n255\n";
        assert_eq!(header, expected_header);
    }

    #[test]
    fn test_constructing_the_ppm_pixel_data() {
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
    fn test_splitting_long_lines_in_ppm_files() {
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
    fn test_ppm_files_are_terminated_by_a_newline_char() {
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
