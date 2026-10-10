//! Phong surface properties, procedural patterns, reflection, and refraction.

use crate::{EPSILON, rendering::canvas::Color, scene::pattern::Pattern};

/// Phong lighting properties and recursive reflection/refraction settings.
///
/// Use struct update syntax to override selected defaults:
///
/// ```
/// use merlin_rt::{rendering::canvas::Color, scene::material::Material};
///
/// let material = Material {
///     color: Color::new(0.2, 0.6, 0.9),
///     reflective: 0.3,
///     ..Material::default()
/// };
/// assert_eq!(material.refractive_index, 1.0);
/// ```
///
/// Coefficients are not validated. Equality compares only color, ambient,
/// diffuse, specular, and shininess, with a floating-point tolerance.
#[derive(Debug)]
pub struct Material {
    /// Base surface color, used when no pattern is set; defaults to white.
    pub color: Color,
    /// Ambient-light coefficient; defaults to `0.1`.
    pub ambient: f64,
    /// Diffuse-light coefficient; defaults to `0.9`.
    pub diffuse: f64,
    /// Specular-highlight coefficient; defaults to `0.9`.
    pub specular: f64,
    /// Specular exponent; larger values give smaller highlights. Defaults to `200.0`.
    pub shininess: f64,
    /// Reflection weight, conventionally `0.0..=1.0`; defaults to `0.0`.
    pub reflective: f64,
    /// Refraction weight, conventionally `0.0..=1.0`; defaults to `0.0` (opaque).
    pub transparency: f64,
    /// Positive optical index; defaults to `1.0` (air). Glass is typically `1.5`.
    pub refractive_index: f64,
    /// Optional procedural color source overriding `color`; defaults to `None`.
    pub pattern: Option<Box<dyn Pattern>>,
}

impl Material {
    /// Creates a material with all properties explicitly supplied, without validation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        color: Color,
        ambient: f64,
        diffuse: f64,
        specular: f64,
        shininess: f64,
        reflective: f64,
        transparency: f64,
        refractive_index: f64,
        pattern: Option<Box<dyn Pattern>>,
    ) -> Self {
        Material {
            color,
            ambient,
            diffuse,
            specular,
            shininess,
            reflective,
            transparency,
            refractive_index,
            pattern,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Material {
            color: Color::new(1.0, 1.0, 1.0),
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.0,
            reflective: 0.0,
            transparency: 0.0,
            refractive_index: 1.0,
            pattern: None,
        }
    }
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.color == other.color
            && (self.ambient - other.ambient).abs() < EPSILON
            && (self.diffuse - other.diffuse).abs() < EPSILON
            && (self.specular - other.specular).abs() < EPSILON
            && (self.shininess - other.shininess).abs() < EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_default_material() {
        let material = Material::default();

        assert_abs_diff_eq!(material.color.r(), 1.0);
        assert_abs_diff_eq!(material.color.g(), 1.0);
        assert_abs_diff_eq!(material.color.b(), 1.0);
        assert_abs_diff_eq!(material.ambient, 0.1);
        assert_abs_diff_eq!(material.diffuse, 0.9);
        assert_abs_diff_eq!(material.specular, 0.9);
        assert_abs_diff_eq!(material.shininess, 200.0);
        assert_abs_diff_eq!(material.reflective, 0.0);
        assert_abs_diff_eq!(material.transparency, 0.0);
        assert_abs_diff_eq!(material.refractive_index, 1.0);
    }
}
