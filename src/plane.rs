use crate::{
    light::Lighting,
    ray::Ray,
    shape::{Intersection, Shape},
    transforms::TransformError,
};

use nalgebra::{Point3, Vector3};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Plane;

impl Shape for Plane {
    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<Self>>, TransformError> {
        let ray = Ray::new(
            self.transform().apply_inverse(ray.origin)?,
            self.transform().apply_inverse(ray.direction)?,
        );

        const EPSILON: f64 = 1e-6;
        if ray.direction.y.abs() < EPSILON {
            Ok(vec![])
        } else {
            let t = -ray.origin.y / ray.direction.y;
            Ok(vec![Intersection::new(t, *self)])
        }
    }

    fn normal_at(&self, _world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        Ok(Vector3::new(0.0, 1.0, 0.0))
    }
}

impl Lighting for Plane {
    fn lighting(
        &self,
        _light: crate::light::PointLight,
        _point: Point3<f64>,
        _eyev: Vector3<f64>,
        _normalv: Vector3<f64>,
        _in_shadow: bool,
    ) -> crate::canvas::Color {
        // For simplicity, we can return a default color for the plane.
        crate::canvas::Color::new(1.0, 1.0, 1.0) // White color
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_normal_of_a_plane_is_constant_everywhere() {
        let plane = Plane {};
        let n1 = plane.normal_at(Point3::new(0.0, 0.0, 0.0)).unwrap();
        let n2 = plane.normal_at(Point3::new(10.0, 0.0, -10.0)).unwrap();
        let n3 = plane.normal_at(Point3::new(-5.0, 0.0, 150.0)).unwrap();

        assert_eq!(n1, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(n2, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(n3, Vector3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_intersect_with_a_ray_parallel_to_the_plane() {
        let plane = Plane {};
        let ray = Ray::new(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn test_intersect_with_a_coplanar_ray() {
        let plane = Plane {};
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn test_a_ray_intersecting_a_plane_from_above() {
        let plane = Plane {};
        let ray = Ray::new(Point3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0].t, 1.0);
        assert_eq!(xs[0].object, plane);
    }

    #[test]
    fn test_a_ray_intersecting_a_plane_from_below() {
        let plane = Plane {};
        let ray = Ray::new(Point3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0].t, 1.0);
        assert_eq!(xs[0].object, plane);
    }
}
