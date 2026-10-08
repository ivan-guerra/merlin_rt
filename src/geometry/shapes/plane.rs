use crate::{
    geometry::{
        ray::Ray,
        shapes::{Intersection, Shape},
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};

use nalgebra::{Point3, Vector3};

#[derive(Debug, PartialEq)]
pub struct Plane {
    pub transform: Transform,
    pub material: Material,
}

impl Default for Plane {
    fn default() -> Self {
        Self {
            transform: Transform::identity(),
            material: Material::default(),
        }
    }
}

impl Shape for Plane {
    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        let ray = Ray::new(
            self.transform().apply_inverse(ray.origin)?,
            self.transform().apply_inverse(ray.direction)?,
        );

        const EPSILON: f64 = 1e-6;
        if ray.direction.y.abs() < EPSILON {
            Ok(vec![])
        } else {
            let t = -ray.origin.y / ray.direction.y;
            Ok(vec![Intersection::new(t, self)])
        }
    }

    fn normal_at(&self, _world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        Ok(self
            .transform
            .apply_transpose_inverse(Vector3::new(0.0, 1.0, 0.0))?
            .normalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_normal_of_a_plane_is_constant_everywhere() {
        let plane = Plane::default();
        let n1 = plane.normal_at(Point3::new(0.0, 0.0, 0.0)).unwrap();
        let n2 = plane.normal_at(Point3::new(10.0, 0.0, -10.0)).unwrap();
        let n3 = plane.normal_at(Point3::new(-5.0, 0.0, 150.0)).unwrap();

        assert_eq!(n1, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(n2, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(n3, Vector3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_intersect_with_a_ray_parallel_to_the_plane() {
        let plane = Plane::default();
        let ray = Ray::new(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn test_intersect_with_a_coplanar_ray() {
        let plane = Plane::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn test_a_ray_intersecting_a_plane_from_above() {
        let plane = Plane::default();
        let ray = Ray::new(Point3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0].t, 1.0);
        assert!(std::ptr::addr_eq(xs[0].object, &plane as &dyn Shape));
    }

    #[test]
    fn test_a_ray_intersecting_a_plane_from_below() {
        let plane = Plane::default();
        let ray = Ray::new(Point3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        let xs = plane.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0].t, 1.0);
        assert!(std::ptr::addr_eq(xs[0].object, &plane as &dyn Shape));
    }
}
