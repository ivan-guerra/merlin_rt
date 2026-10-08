use crate::{
    geometry::{
        ray::Ray,
        shapes::{Intersection, Shape},
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};

use approx::abs_diff_eq;
use nalgebra::{Point3, Vector3};

const EPSILON: f64 = 1e-6;

#[derive(Debug)]
pub struct DoubleNappedCone {
    pub minimum: f64,
    pub maximum: f64,
    pub closed: bool,
    pub transform: Transform,
    pub material: Material,
}

impl DoubleNappedCone {
    pub fn new(
        minimum: f64,
        maximum: f64,
        closed: bool,
        transform: Transform,
        material: Material,
    ) -> Self {
        Self {
            minimum,
            maximum,
            closed,
            transform,
            material,
        }
    }

    fn check_cap(&self, ray: &Ray, t: f64, radius: f64) -> bool {
        let x = ray.origin.x + t * ray.direction.x;
        let z = ray.origin.z + t * ray.direction.z;

        x * x + z * z <= radius * radius
    }

    fn intersect_caps(&self, ray: &Ray) -> Vec<Intersection<'_>> {
        if !self.closed || ray.direction.y.abs() < EPSILON {
            return Vec::new();
        }

        let mut intersections = Vec::new();

        let minimum_t = (self.minimum - ray.origin.y) / ray.direction.y;
        if self.check_cap(ray, minimum_t, self.minimum.abs()) {
            intersections.push(Intersection::new(minimum_t, self));
        }

        let maximum_t = (self.maximum - ray.origin.y) / ray.direction.y;
        if self.check_cap(ray, maximum_t, self.maximum.abs()) {
            intersections.push(Intersection::new(maximum_t, self));
        }

        intersections
    }
}

impl Default for DoubleNappedCone {
    fn default() -> Self {
        Self {
            minimum: f64::NEG_INFINITY,
            maximum: f64::INFINITY,
            closed: false,
            transform: Transform::identity(),
            material: Material::default(),
        }
    }
}

impl Shape for DoubleNappedCone {
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
            self.transform.apply_inverse(ray.origin)?,
            self.transform.apply_inverse(ray.direction)?,
        );

        let a = ray.direction.x.powi(2) - ray.direction.y.powi(2) + ray.direction.z.powi(2);
        let b = 2.0
            * (ray.origin.x * ray.direction.x - ray.origin.y * ray.direction.y
                + ray.origin.z * ray.direction.z);
        let c = ray.origin.x.powi(2) - ray.origin.y.powi(2) + ray.origin.z.powi(2);

        let mut intersections = Vec::new();

        if a.abs() < EPSILON {
            if b.abs() >= EPSILON {
                let t = -c / (2.0 * b);
                let y = ray.origin.y + t * ray.direction.y;

                if self.minimum < y && y < self.maximum {
                    intersections.push(Intersection::new(t, self));
                }
            }
        } else {
            let discriminant = b.powi(2) - 4.0 * a * c;

            if discriminant >= 0.0 {
                let mut t0 = (-b - discriminant.sqrt()) / (2.0 * a);
                let mut t1 = (-b + discriminant.sqrt()) / (2.0 * a);

                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }

                let y0 = ray.origin.y + t0 * ray.direction.y;
                if self.minimum < y0 && y0 < self.maximum {
                    intersections.push(Intersection::new(t0, self));
                }

                let y1 = ray.origin.y + t1 * ray.direction.y;
                if self.minimum < y1 && y1 < self.maximum {
                    intersections.push(Intersection::new(t1, self));
                }
            }
        }

        intersections.extend(self.intersect_caps(&ray));
        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));

        Ok(intersections)
    }

    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        let object_point = self.transform.apply_inverse(world_point)?;
        let distance = object_point.x.powi(2) + object_point.z.powi(2);

        let object_normal = if self.closed
            && distance <= self.maximum.abs().powi(2)
            && object_point.y >= self.maximum - EPSILON
        {
            Vector3::new(0.0, 1.0, 0.0)
        } else if self.closed
            && distance <= self.minimum.abs().powi(2)
            && object_point.y <= self.minimum + EPSILON
        {
            Vector3::new(0.0, -1.0, 0.0)
        } else {
            let y = distance.sqrt();

            if object_point.y > 0.0 {
                Vector3::new(object_point.x, -y, object_point.z)
            } else {
                Vector3::new(object_point.x, y, object_point.z)
            }
        };
        let world_normal = self.transform.apply_transpose_inverse(object_normal)?;

        if abs_diff_eq!(world_normal.norm(), 0.0) {
            Ok(world_normal)
        } else {
            Ok(world_normal.normalize())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_intersecting_a_cone_with_a_ray() {
        let cone = DoubleNappedCone::default();
        let rays = [
            Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(
                Point3::new(0.0, 0.0, -5.0),
                Vector3::new(1.0, 1.0, 1.0).normalize(),
            ),
            Ray::new(
                Point3::new(1.0, 1.0, -5.0),
                Vector3::new(-0.5, -1.0, 1.0).normalize(),
            ),
        ];
        let expected_ts = [(5.0, 5.0), (8.66025, 8.66025), (4.55006, 49.44994)];

        for (ray, (expected_t0, expected_t1)) in rays.iter().zip(expected_ts.iter()) {
            let xs = cone.intersect(ray).unwrap();
            assert_eq!(xs.len(), 2);
            assert_abs_diff_eq!(xs[0].t, *expected_t0, epsilon = 1e-5);
            assert_abs_diff_eq!(xs[1].t, *expected_t1, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_intersecting_a_cone_with_a_ray_parallel_to_one_of_its_halves() {
        let cone = DoubleNappedCone::default();
        let ray = Ray::new(
            Point3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 1.0).normalize(),
        );
        let xs = cone.intersect(&ray).unwrap();

        assert_eq!(xs.len(), 1);
        assert_abs_diff_eq!(xs[0].t, 0.35355, epsilon = 1e-5);
    }

    #[test]
    fn test_intersecting_a_cone_end_caps() {
        let cone = DoubleNappedCone {
            minimum: -0.5,
            maximum: 0.5,
            closed: true,
            ..Default::default()
        };

        let rays = [
            Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 1.0, 0.0)),
            Ray::new(
                Point3::new(0.0, 0.0, -0.25),
                Vector3::new(0.0, 1.0, 1.0).normalize(),
            ),
            Ray::new(Point3::new(0.0, 0.0, -0.25), Vector3::new(0.0, 1.0, 0.0)),
        ];
        let expected_counts = [0, 2, 4];

        for (ray, expected_count) in rays.iter().zip(expected_counts.iter()) {
            let xs = cone.intersect(ray).unwrap();
            assert_eq!(xs.len(), *expected_count);
        }
    }

    #[test]
    fn test_computing_the_normal_vector_on_a_cone() {
        let cone = DoubleNappedCone::default();
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, -1.0, 0.0),
        ];
        let expected_normals = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, -2f64.sqrt(), 1.0).normalize(),
            Vector3::new(-1.0, 1.0, 0.0).normalize(),
        ];

        for (point, expected_normal) in points.iter().zip(expected_normals.iter()) {
            let normal = cone.normal_at(*point).unwrap();
            assert_abs_diff_eq!(normal.x, expected_normal.x);
            assert_abs_diff_eq!(normal.y, expected_normal.y);
            assert_abs_diff_eq!(normal.z, expected_normal.z);
        }
    }
}
