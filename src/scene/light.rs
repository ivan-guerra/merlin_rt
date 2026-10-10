//! Point lights with a position and RGB intensity.

use crate::rendering::canvas::Color;

use nalgebra::Point3;

/// A point light with RGB intensity and no distance attenuation.
#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    /// Light position in world space.
    pub position: Point3<f64>,
    /// Light color and brightness; `(1, 1, 1)` is white at unit intensity.
    pub intensity: Color,
}

impl PointLight {
    /// Creates a light at a world-space position with the given intensity.
    pub fn new(position: Point3<f64>, intensity: Color) -> Self {
        PointLight {
            position,
            intensity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_light_has_position_and_intensity() {
        let position = Point3::new(1.0, 2.0, 3.0);
        let intensity = Color::new(1.0, 1.0, 1.0);
        let light = PointLight::new(position, intensity);

        assert_eq!(light.position, position);
        assert_eq!(light.intensity, intensity);
    }
}
