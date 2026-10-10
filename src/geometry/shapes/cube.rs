use crate::{
    EPSILON,
    geometry::{
        ray::Ray,
        shapes::{Intersection, ParentLink, Shape},
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};

use nalgebra::{Point3, Vector3};

fn check_axis(origin: f64, direction: f64) -> (f64, f64) {
    let tmin_numerator = -1.0 - origin;
    let tmax_numerator = 1.0 - origin;

    let (tmin, tmax) = if direction.abs() >= EPSILON {
        (tmin_numerator / direction, tmax_numerator / direction)
    } else {
        (
            tmin_numerator * f64::INFINITY,
            tmax_numerator * f64::INFINITY,
        )
    };

    if tmin > tmax {
        (tmax, tmin)
    } else {
        (tmin, tmax)
    }
}

/// An axis-aligned cube spanning `-1..=1` on each object-space axis.
///
/// Defaults to an identity transform and default material.
#[derive(Debug)]
pub struct Cube {
    parent: ParentLink,
    material: Material,
    transform: Transform,
}

impl Default for Cube {
    fn default() -> Self {
        Self {
            material: Material::default(),
            parent: ParentLink::default(),
            transform: Transform::identity(),
        }
    }
}

impl Cube {
    /// Creates a builder initialized with [`Cube::default`].
    pub fn builder() -> CubeBuilder {
        CubeBuilder::default()
    }

    /// Creates a cube with the supplied material and object-to-parent transform.
    pub fn new(material: Material, transform: Transform) -> Self {
        Self::builder()
            .transform(transform)
            .material(material)
            .build()
    }
}

/// A builder for [`Cube`]; see the shape for its defaults.
#[derive(Debug, Default)]
#[must_use = "call build() to create the shape"]
pub struct CubeBuilder {
    shape: Cube,
}

impl CubeBuilder {
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
    pub fn build(self) -> Cube {
        self.shape
    }
}

impl Shape for Cube {
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
        let ray = Ray::new(
            self.transform().apply_inverse(ray.origin)?,
            self.transform().apply_inverse(ray.direction)?,
        );

        let (xtmin, xtmax) = check_axis(ray.origin.x, ray.direction.x);
        let (ytmin, ytmax) = check_axis(ray.origin.y, ray.direction.y);
        let (ztmin, ztmax) = check_axis(ray.origin.z, ray.direction.z);

        let tmin = xtmin.max(ytmin).max(ztmin);
        let tmax = xtmax.min(ytmax).min(ztmax);

        if tmin > tmax {
            Ok(vec![])
        } else {
            Ok(vec![
                Intersection::new(tmin, self),
                Intersection::new(tmax, self),
            ])
        }
    }

    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        let object_point = self.world_to_object(world_point)?;
        let maxc = object_point
            .x
            .abs()
            .max(object_point.y.abs())
            .max(object_point.z.abs());

        let object_normal = if maxc == object_point.x.abs() {
            Vector3::new(object_point.x, 0.0, 0.0)
        } else if maxc == object_point.y.abs() {
            Vector3::new(0.0, object_point.y, 0.0)
        } else {
            Vector3::new(0.0, 0.0, object_point.z)
        };

        self.normal_to_world(object_normal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_a_ray_intersects_a_cube() {
        let c = Cube::default();
        let rays = [
            Ray::new(Point3::new(5.0, 0.5, 0.0), Vector3::new(-1.0, 0.0, 0.0)),
            Ray::new(Point3::new(-5.0, 0.5, 0.0), Vector3::new(1.0, 0.0, 0.0)),
            Ray::new(Point3::new(0.5, 5.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
            Ray::new(Point3::new(0.5, -5.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            Ray::new(Point3::new(0.5, 0.0, 5.0), Vector3::new(0.0, 0.0, -1.0)),
            Ray::new(Point3::new(0.5, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(Point3::new(0.0, 0.5, 0.0), Vector3::new(0.0, 0.0, 1.0)),
        ];
        let expected_ts = [
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (-1.0, 1.0),
        ];

        for (ray, (expected_t1, expected_t2)) in rays.iter().zip(expected_ts.iter()) {
            let xs = c.intersect(ray).unwrap();
            assert_eq!(xs.len(), 2);
            assert_abs_diff_eq!(xs[0].t, *expected_t1);
            assert_abs_diff_eq!(xs[1].t, *expected_t2);
        }
    }

    #[test]
    fn test_the_normal_on_the_surface_of_a_cube() {
        let c = Cube::default();
        let points = [
            Point3::new(1.0, 0.5, -0.8),
            Point3::new(-1.0, -0.2, 0.9),
            Point3::new(-0.4, 1.0, -0.1),
            Point3::new(0.3, -1.0, -0.7),
            Point3::new(-0.6, 0.3, 1.0),
            Point3::new(0.4, 0.4, -1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, -1.0, -1.0),
        ];
        let expected_normals = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];

        for (point, expected_normal) in points.iter().zip(expected_normals.iter()) {
            let normal = c.normal_at(*point).unwrap();
            assert_abs_diff_eq!(normal.x, expected_normal.x);
            assert_abs_diff_eq!(normal.y, expected_normal.y);
            assert_abs_diff_eq!(normal.z, expected_normal.z);
        }
    }
}
