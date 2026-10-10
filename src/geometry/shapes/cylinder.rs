use crate::{
    EPSILON,
    geometry::{
        ray::Ray,
        shapes::{Intersection, ParentLink, Shape},
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};

use approx::abs_diff_eq;
use nalgebra::{Point3, Vector3};

/// A unit-radius cylinder around the object-space Y axis.
///
/// Defaults to infinite Y bounds, open ends, an identity transform, and a
/// default material. Set finite bounds before enabling end caps.
#[derive(Debug)]
pub struct Cylinder {
    parent: ParentLink,
    minimum: f64,
    maximum: f64,
    closed: bool,
    transform: Transform,
    material: Material,
}

impl Cylinder {
    /// Creates a builder initialized with [`Cylinder::default`].
    pub fn builder() -> CylinderBuilder {
        CylinderBuilder::default()
    }

    /// Creates a shape with explicit Y bounds, cap state, transform, and material.
    ///
    /// Use `minimum < maximum` and finite bounds for a closed shape. Parameters
    /// are stored without validation.
    pub fn new(
        minimum: f64,
        maximum: f64,
        closed: bool,
        transform: Transform,
        material: Material,
    ) -> Self {
        Self::builder()
            .minimum(minimum)
            .maximum(maximum)
            .closed(closed)
            .transform(transform)
            .material(material)
            .build()
    }

    fn intersect_caps(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        let mut xs = vec![];

        if !self.closed || abs_diff_eq!(ray.direction.y, 0.0) {
            return Ok(xs);
        }

        let t0 = (self.minimum - ray.origin.y) / ray.direction.y;
        if self.check_cap(ray, t0) {
            xs.push(Intersection::new(t0, self));
        }

        let t1 = (self.maximum - ray.origin.y) / ray.direction.y;
        if self.check_cap(ray, t1) {
            xs.push(Intersection::new(t1, self));
        }

        Ok(xs)
    }

    fn check_cap(&self, ray: &Ray, t: f64) -> bool {
        let x = ray.origin.x + t * ray.direction.x;
        let z = ray.origin.z + t * ray.direction.z;
        (x.powi(2) + z.powi(2)) <= 1.0
    }
}

impl Default for Cylinder {
    fn default() -> Self {
        Self {
            minimum: f64::NEG_INFINITY,
            maximum: f64::INFINITY,
            closed: false,
            parent: ParentLink::default(),
            transform: Transform::identity(),
            material: Material::default(),
        }
    }
}

/// A builder for [`Cylinder`]; see the shape for its defaults.
#[derive(Debug, Default)]
#[must_use = "call build() to create the shape"]
pub struct CylinderBuilder {
    shape: Cylinder,
}

impl CylinderBuilder {
    /// Sets the lower object-space Y bound; defaults to negative infinity.
    pub fn minimum(mut self, minimum: f64) -> Self {
        self.shape.minimum = minimum;
        self
    }

    /// Sets the upper object-space Y bound; defaults to positive infinity.
    pub fn maximum(mut self, maximum: f64) -> Self {
        self.shape.maximum = maximum;
        self
    }

    /// Enables or disables end caps; defaults to `false`.
    ///
    /// Use finite Y bounds when enabling caps.
    pub fn closed(mut self, closed: bool) -> Self {
        self.shape.closed = closed;
        self
    }

    /// Sets the object-to-parent transform, replacing any previous transform.
    pub fn transform(mut self, transform: Transform) -> Self {
        self.shape.transform = transform;
        self
    }

    /// Sets the surface material.
    pub fn material(mut self, material: Material) -> Self {
        self.shape.material = material;
        self
    }

    /// Returns the configured shape.
    pub fn build(self) -> Cylinder {
        self.shape
    }
}

impl Shape for Cylinder {
    fn parent_link(&self) -> &ParentLink {
        &self.parent
    }

    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn intersect(&self, _ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        let ray = Ray::new(
            self.transform().apply_inverse(_ray.origin)?,
            self.transform().apply_inverse(_ray.direction)?,
        );

        let a = ray.direction.x.powi(2) + ray.direction.z.powi(2);
        if abs_diff_eq!(a, 0.0) {
            return self.intersect_caps(&ray);
        }

        let b = (2.0 * ray.origin.x * ray.direction.x) + (2.0 * ray.origin.z * ray.direction.z);
        let c = ray.origin.x.powi(2) + ray.origin.z.powi(2) - 1.0;
        let disc = b.powi(2) - 4.0 * a * c;

        if disc < 0.0 {
            return Ok(vec![]);
        }

        let mut t0 = (-b - disc.sqrt()) / (2.0 * a);
        let mut t1 = (-b + disc.sqrt()) / (2.0 * a);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }

        let mut xs = vec![];

        let y0 = ray.origin.y + t0 * ray.direction.y;
        if self.minimum < y0 && y0 < self.maximum {
            xs.push(Intersection::new(t0, self));
        }

        let y1 = ray.origin.y + t1 * ray.direction.y;
        if self.minimum < y1 && y1 < self.maximum {
            xs.push(Intersection::new(t1, self));
        }

        xs.extend(self.intersect_caps(&ray)?);
        xs.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

        Ok(xs)
    }

    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        let object_point = self.world_to_object(world_point)?;
        let dist = object_point.x.powi(2) + object_point.z.powi(2);

        if dist < 1.0 && object_point.y >= self.maximum - EPSILON {
            self.normal_to_world(Vector3::new(0.0, 1.0, 0.0))
        } else if dist < 1.0 && object_point.y <= self.minimum + EPSILON {
            self.normal_to_world(Vector3::new(0.0, -1.0, 0.0))
        } else {
            let object_normal = Vector3::new(object_point.x, 0.0, object_point.z);
            self.normal_to_world(object_normal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_a_ray_misses_a_cylinder() {
        let cyl = Cylinder::default();
        let ray = Ray::new(Point3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        let xs = cyl.intersect(&ray).unwrap();
        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn test_a_ray_strikes_a_cylinder() {
        let cyl = Cylinder::default();

        let rays = [
            Ray::new(Point3::new(1.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(
                Point3::new(0.5, 0.0, -5.0),
                Vector3::new(0.1, 1.0, 1.0).normalize(),
            ),
        ];
        let expected_ts = [(5.0, 5.0), (4.0, 6.0), (6.80798, 7.08872)];

        for (ray, (expected_t0, expected_t1)) in rays.iter().zip(expected_ts.iter()) {
            let xs = cyl.intersect(ray).unwrap();
            assert_eq!(xs.len(), 2);
            assert_abs_diff_eq!(xs[0].t, *expected_t0, epsilon = EPSILON);
            assert_abs_diff_eq!(xs[1].t, *expected_t1, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_normal_vector_on_a_cylinder() {
        let cyl = Cylinder::default();

        let points = [
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 5.0, -1.0),
            Point3::new(0.0, -2.0, 1.0),
            Point3::new(-1.0, 1.0, 0.0),
        ];
        let expected_normals = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];

        for (point, expected_normal) in points.iter().zip(expected_normals.iter()) {
            let normal = cyl.normal_at(*point).unwrap();
            assert_abs_diff_eq!(normal.x, expected_normal.x);
            assert_abs_diff_eq!(normal.y, expected_normal.y);
            assert_abs_diff_eq!(normal.z, expected_normal.z);
        }
    }

    #[test]
    fn test_intersecting_a_constrained_cylinder() {
        let cyl = Cylinder::builder().minimum(1.0).maximum(2.0).build();

        let rays = [
            Ray::new(
                Point3::new(0.0, 1.5, 0.0),
                Vector3::new(0.1, 1.0, 0.0).normalize(),
            ),
            Ray::new(Point3::new(0.0, 3.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(Point3::new(0.0, 2.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(Point3::new(0.0, 1.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(Point3::new(0.0, 1.5, -2.0), Vector3::new(0.0, 0.0, 1.0)),
        ];
        let expected_counts = [0, 0, 0, 0, 0, 2];

        for (ray, expected_count) in rays.iter().zip(expected_counts.iter()) {
            let xs = cyl.intersect(ray).unwrap();
            assert_eq!(xs.len(), *expected_count);
        }
    }

    #[test]
    fn test_intersecting_the_caps_of_a_closed_cylinder() {
        let cyl = Cylinder::builder()
            .minimum(1.0)
            .maximum(2.0)
            .closed(true)
            .build();

        let rays = [
            Ray::new(Point3::new(0.0, 3.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
            Ray::new(
                Point3::new(0.0, 3.0, -2.0),
                Vector3::new(0.0, -1.0, 2.0).normalize(),
            ),
            Ray::new(
                Point3::new(0.0, 4.0, -2.0),
                Vector3::new(0.0, -1.0, 1.0).normalize(),
            ),
            Ray::new(
                Point3::new(0.0, 0.0, -2.0),
                Vector3::new(0.0, 1.0, 2.0).normalize(),
            ),
            Ray::new(
                Point3::new(0.0, -1.0, -2.0),
                Vector3::new(0.0, 1.0, 1.0).normalize(),
            ),
        ];
        let expected_counts = [2, 2, 2, 2, 2];

        for (ray, expected_count) in rays.iter().zip(expected_counts.iter()) {
            let xs = cyl.intersect(ray).unwrap();
            assert_eq!(xs.len(), *expected_count);
        }
    }

    #[test]
    fn test_the_normal_vector_on_a_cylinder_end_caps() {
        let cyl = Cylinder::builder()
            .minimum(1.0)
            .maximum(2.0)
            .closed(true)
            .build();

        let points = [
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.5),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(0.5, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.5),
        ];
        let expected_normals = [
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];

        for (point, expected_normal) in points.iter().zip(expected_normals.iter()) {
            let normal = cyl.normal_at(*point).unwrap();
            assert_abs_diff_eq!(normal.x, expected_normal.x);
            assert_abs_diff_eq!(normal.y, expected_normal.y);
            assert_abs_diff_eq!(normal.z, expected_normal.z);
        }
    }
}
