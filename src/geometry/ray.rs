//! Parametric rays used for intersections and secondary light paths.

use crate::geometry::transforms::Transform;

use nalgebra::{Point3, Vector3};

/// A parametric ray: `origin + direction * t`.
///
/// The direction is not normalized automatically; `t` is a distance only when
/// the direction has unit length.
#[derive(Debug, Clone)]
pub struct Ray {
    /// Starting point of the ray.
    pub origin: Point3<f64>,
    /// Direction and parameter scale of the ray.
    pub direction: Vector3<f64>,
}

impl Ray {
    /// Creates a ray without normalizing its direction.
    pub fn new(origin: Point3<f64>, direction: Vector3<f64>) -> Self {
        Ray { origin, direction }
    }

    /// Returns the point at parameter `t`, which may be negative.
    pub fn position(&self, t: f64) -> Point3<f64> {
        self.origin + (self.direction * t)
    }

    /// Transforms the origin as a point and the direction as a vector.
    ///
    /// Translation affects only the origin; the direction is not renormalized.
    pub fn transform(&self, transform: &Transform) -> Self {
        Self {
            origin: transform.apply(self.origin),
            direction: transform.apply(self.direction),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::transforms::Axis;
    use approx::assert_abs_diff_eq;
    use nalgebra::{Point3, Scale3, Translation3, Vector3};

    #[test]
    fn test_creating_and_querying_a_ray() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let direction = Vector3::new(4.0, 5.0, 6.0);
        let ray = Ray::new(origin, direction);

        assert_eq!(ray.origin, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(ray.direction, Vector3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_computing_a_point_from_a_distance() {
        let ray = Ray::new(Point3::new(2.0, 3.0, 4.0), Vector3::new(1.0, 0.0, 0.0));

        assert_eq!(ray.position(0.0), Point3::new(2.0, 3.0, 4.0));
        assert_eq!(ray.position(1.0), Point3::new(3.0, 3.0, 4.0));
        assert_eq!(ray.position(-1.0), Point3::new(1.0, 3.0, 4.0));
        assert_eq!(ray.position(2.5), Point3::new(4.5, 3.0, 4.0));
    }

    #[test]
    fn test_translating_a_ray() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 1.0, 0.0));
        let transform = Transform::translation(Translation3::new(3.0, 4.0, 5.0));
        let transformed_ray = ray.transform(&transform);

        assert_eq!(transformed_ray.origin, Point3::new(4.0, 6.0, 8.0));
        assert_eq!(transformed_ray.direction, Vector3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_scaling_a_ray() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 1.0, 0.0));
        let transform = Transform::scale(Scale3::new(2.0, 3.0, 4.0));
        let transformed_ray = ray.transform(&transform);

        assert_eq!(transformed_ray.origin, Point3::new(2.0, 6.0, 12.0));
        assert_eq!(transformed_ray.direction, Vector3::new(0.0, 3.0, 0.0));
    }

    #[test]
    fn test_rotating_a_ray() {
        let ray = Ray::new(Point3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        let transform = Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2);
        let transformed_ray = ray.transform(&transform);

        assert_abs_diff_eq!(transformed_ray.origin.x, 0.0);
        assert_abs_diff_eq!(transformed_ray.origin.y, 0.0);
        assert_abs_diff_eq!(transformed_ray.origin.z, 1.0);
        assert_abs_diff_eq!(transformed_ray.direction.x, 0.0);
        assert_abs_diff_eq!(transformed_ray.direction.y, 0.0);
        assert_abs_diff_eq!(transformed_ray.direction.z, 1.0);
    }
}
