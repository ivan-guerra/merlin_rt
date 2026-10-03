use crate::{canvas::Color, pattern::Pattern};

#[derive(Debug)]
pub struct Material {
    pub color: Color,
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub shininess: f64,
    pub pattern: Option<Box<dyn Pattern>>,
}

impl Material {
    pub fn new(
        color: Color,
        ambient: f64,
        diffuse: f64,
        specular: f64,
        shininess: f64,
        pattern: Option<Box<dyn Pattern>>,
    ) -> Self {
        Material {
            color,
            ambient,
            diffuse,
            specular,
            shininess,
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
            pattern: None,
        }
    }
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.color == other.color
            && (self.ambient - other.ambient).abs() < 1e-5
            && (self.diffuse - other.diffuse).abs() < 1e-5
            && (self.specular - other.specular).abs() < 1e-5
            && (self.shininess - other.shininess).abs() < 1e-5
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
    }
}
