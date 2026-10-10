use crate::{
    geometry::{
        ray::Ray,
        shapes::{Intersection, ParentLink, Shape},
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};

use nalgebra::{Point3, Vector3};

/// A flat triangle with object-space vertices and cached edges and normal.
/// Degenerate triangles have a zero normal and produce no intersections.
#[derive(Debug, PartialEq)]
pub struct Triangle {
    parent: ParentLink,
    transform: Transform,
    material: Material,
    p1: Point3<f64>,
    p2: Point3<f64>,
    p3: Point3<f64>,
    e1: Vector3<f64>,
    e2: Vector3<f64>,
    normal: Vector3<f64>,
}

impl Triangle {
    pub fn builder() -> TriangleBuilder {
        TriangleBuilder::default()
    }

    pub fn new(p1: Point3<f64>, p2: Point3<f64>, p3: Point3<f64>) -> Self {
        let e1 = p2 - p1;
        let e2 = p3 - p1;
        let normal = e2
            .cross(&e1)
            .try_normalize(0.0)
            .unwrap_or_else(Vector3::zeros);

        Self {
            parent: ParentLink::default(),
            transform: Transform::identity(),
            material: Material::default(),
            p1,
            p2,
            p3,
            e1,
            e2,
            normal,
        }
    }

    pub fn p1(&self) -> Point3<f64> {
        self.p1
    }

    pub fn p2(&self) -> Point3<f64> {
        self.p2
    }

    pub fn p3(&self) -> Point3<f64> {
        self.p3
    }

    pub fn e1(&self) -> Vector3<f64> {
        self.e1
    }

    pub fn e2(&self) -> Vector3<f64> {
        self.e2
    }

    pub fn normal(&self) -> Vector3<f64> {
        self.normal
    }

    /// Shared Möller–Trumbore intersection for flat and smooth triangles.
    /// The ray is in parent space; the returned weights belong to p2 and p3.
    pub(super) fn intersect_uv(
        &self,
        ray: &Ray,
    ) -> Result<Option<(f64, f64, f64)>, TransformError> {
        let ray = Ray::new(
            self.transform().apply_inverse(ray.origin)?,
            self.transform().apply_inverse(ray.direction)?,
        );

        const EPSILON: f64 = 1e-6;
        let direction_cross_e2 = ray.direction.cross(&self.e2);
        let determinant = self.e1.dot(&direction_cross_e2);

        if determinant.abs() < EPSILON {
            return Ok(None);
        }

        let f = 1.0 / determinant;
        let p1_to_origin = ray.origin - self.p1;
        let u = f * p1_to_origin.dot(&direction_cross_e2);

        if !(0.0..=1.0).contains(&u) {
            return Ok(None);
        }

        let origin_cross_e1 = p1_to_origin.cross(&self.e1);
        let v = f * ray.direction.dot(&origin_cross_e1);

        if v < 0.0 || u + v > 1.0 {
            return Ok(None);
        }

        let t = f * self.e2.dot(&origin_cross_e1);
        Ok(Some((t, u, v)))
    }
}

impl Default for Triangle {
    fn default() -> Self {
        Self::new(
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        )
    }
}

#[derive(Debug, Default)]
#[must_use = "call build() to create the shape"]
pub struct TriangleBuilder {
    shape: Triangle,
}

impl TriangleBuilder {
    pub fn vertices(mut self, p1: Point3<f64>, p2: Point3<f64>, p3: Point3<f64>) -> Self {
        self.shape = Triangle {
            transform: self.shape.transform,
            material: self.shape.material,
            ..Triangle::new(p1, p2, p3)
        };
        self
    }

    pub fn transform(mut self, transform: Transform) -> Self {
        self.shape.transform = transform;
        self
    }

    pub fn material(mut self, material: Material) -> Self {
        self.shape.material = material;
        self
    }

    pub fn build(self) -> Triangle {
        self.shape
    }
}

impl Shape for Triangle {
    fn parent_link(&self) -> &ParentLink {
        &self.parent
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        Ok(self
            .intersect_uv(ray)?
            .map(|(t, _, _)| Intersection::new(t, self))
            .into_iter()
            .collect())
    }

    fn normal_at(&self, _world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        self.normal_to_world(self.normal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{shapes::Group, transforms::Axis};
    use approx::assert_abs_diff_eq;
    use nalgebra::{Scale3, Translation3};
    use std::{f64::consts::FRAC_PI_2, rc::Rc};

    #[test]
    fn test_constructing_a_triangle() {
        let p1 = Point3::new(0.0, 1.0, 0.0);
        let p2 = Point3::new(-1.0, 0.0, 0.0);
        let p3 = Point3::new(1.0, 0.0, 0.0);
        let triangle = Triangle::new(p1, p2, p3);

        assert_eq!(triangle.p1(), p1);
        assert_eq!(triangle.p2(), p2);
        assert_eq!(triangle.p3(), p3);
        assert_eq!(triangle.e1(), Vector3::new(-1.0, -1.0, 0.0));
        assert_eq!(triangle.e2(), Vector3::new(1.0, -1.0, 0.0));
        assert_eq!(triangle.normal(), Vector3::new(0.0, 0.0, -1.0));
        assert_eq!(*triangle.transform(), Transform::identity());
        assert_eq!(*triangle.material(), Material::default());
        assert!(triangle.parent().is_none());
    }

    #[test]
    fn test_the_normal_of_a_triangle_is_constant_everywhere() {
        let triangle = Triangle::default();

        for point in [
            Point3::new(0.0, 0.5, 0.0),
            Point3::new(-0.5, 0.75, 0.0),
            Point3::new(0.5, 0.25, 0.0),
        ] {
            assert_eq!(triangle.normal_at(point).unwrap(), triangle.normal());
        }
    }

    #[test]
    fn test_intersecting_a_ray_parallel_to_the_triangle() {
        let triangle = Triangle::default();
        let ray = Ray::new(Point3::new(0.0, -1.0, -2.0), Vector3::new(0.0, 1.0, 0.0));

        assert!(triangle.intersect(&ray).unwrap().is_empty());
    }

    #[test]
    fn test_a_ray_misses_each_triangle_edge() {
        let triangle = Triangle::default();

        for origin in [
            Point3::new(1.0, 1.0, -2.0),
            Point3::new(-1.0, 1.0, -2.0),
            Point3::new(0.0, -1.0, -2.0),
        ] {
            let ray = Ray::new(origin, Vector3::new(0.0, 0.0, 1.0));
            assert!(triangle.intersect(&ray).unwrap().is_empty());
        }
    }

    #[test]
    fn test_a_ray_strikes_a_triangle() {
        let triangle = Triangle::default();
        let ray = Ray::new(Point3::new(0.0, 0.5, -2.0), Vector3::new(0.0, 0.0, 1.0));
        let xs = triangle.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 1);
        assert_abs_diff_eq!(xs[0].t, 2.0);
        assert!(std::ptr::addr_eq(xs[0].object, &triangle as &dyn Shape));
    }

    #[test]
    fn test_intersections_from_both_sides_and_behind_the_ray() {
        let triangle = Triangle::default();

        for (z, direction_z, expected_t) in [(-2.0, 1.0, 2.0), (2.0, -1.0, 2.0), (2.0, 1.0, -2.0)] {
            let ray = Ray::new(
                Point3::new(0.0, 0.5, z),
                Vector3::new(0.0, 0.0, direction_z),
            );
            let xs = triangle.intersect(&ray).unwrap();

            assert_eq!(xs.len(), 1);
            assert_abs_diff_eq!(xs[0].t, expected_t);
        }
    }

    #[test]
    fn test_building_a_transformed_triangle_with_a_material() {
        let transform = Transform::translation(Translation3::new(0.0, 0.0, 3.0));
        let material = Material {
            ambient: 0.5,
            ..Default::default()
        };
        let triangle = Triangle::builder()
            .transform(transform)
            .material(material)
            .vertices(
                Point3::new(0.0, 2.0, 0.0),
                Point3::new(-2.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
            )
            .build();
        let ray = Ray::new(Point3::new(0.0, 1.0, -2.0), Vector3::new(0.0, 0.0, 1.0));
        let xs = triangle.intersect(&ray).unwrap();

        assert_eq!(*triangle.transform(), transform);
        assert_abs_diff_eq!(triangle.material().ambient, 0.5);
        assert_eq!(triangle.e1(), Vector3::new(-2.0, -2.0, 0.0));
        assert_eq!(xs.len(), 1);
        assert_abs_diff_eq!(xs[0].t, 5.0);
    }

    #[test]
    fn test_intersections_and_normals_in_a_transformed_group() {
        let triangle = Rc::new(
            Triangle::builder()
                .transform(Transform::scale(Scale3::new(1.0, 2.0, 3.0)))
                .build(),
        );
        let group = Group::builder()
            .transform(Transform::rotation(Axis::Y, FRAC_PI_2))
            .child(triangle.clone())
            .build()
            .unwrap();
        let ray = Ray::new(Point3::new(-2.0, 1.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let xs = group.intersect(&ray).unwrap();

        assert!(triangle.parent().is_some());
        assert_eq!(xs.len(), 1);
        assert_abs_diff_eq!(xs[0].t, 2.0, epsilon = 1e-10);
        assert!(std::ptr::addr_eq(
            xs[0].object,
            triangle.as_ref() as &dyn Shape,
        ));
        assert_abs_diff_eq!(
            triangle.normal_at(Point3::new(0.0, 1.0, 0.0)).unwrap(),
            Vector3::new(-1.0, 0.0, 0.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_a_degenerate_triangle_has_no_intersections() {
        let triangle = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        let ray = Ray::new(Point3::new(1.0, 0.0, -2.0), Vector3::new(0.0, 0.0, 1.0));

        assert_eq!(triangle.normal(), Vector3::zeros());
        assert!(triangle.intersect(&ray).unwrap().is_empty());
    }
}
