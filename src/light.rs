use crate::{canvas::Color, transforms::TransformError};

use nalgebra::{Point3, Vector3};

#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub position: Point3<f64>,
    pub intensity: Color,
}

impl PointLight {
    pub fn new(position: Point3<f64>, intensity: Color) -> Self {
        PointLight {
            position,
            intensity,
        }
    }
}

pub trait Lighting {
    fn lighting(
        &self,
        light: PointLight,
        point: Point3<f64>,
        eyev: Vector3<f64>,
        normalv: Vector3<f64>,
        in_shadow: bool,
    ) -> Result<Color, TransformError>;
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
