use crate::transforms::{Transform, TransformError};

use nalgebra::{Point3, Vector3};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Ray {
    pub origin: Point3<f64>,
    pub direction: Vector3<f64>,
}

impl Ray {
    pub fn new(origin: Point3<f64>, direction: Vector3<f64>) -> Self {
        Ray { origin, direction }
    }

    pub fn position(&self, t: f64) -> Point3<f64> {
        self.origin + (self.direction * t)
    }

    pub fn transform(&self, transform: &Transform) -> Self {
        Self {
            origin: transform.apply(self.origin),
            direction: transform.apply(self.direction),
        }
    }
}

pub trait Intersectable: Clone {
    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<Self>>, TransformError>;
}

#[derive(Debug)]
pub struct Intersection<T: Intersectable> {
    pub t: f64,
    pub object: T,
}

impl<T: Intersectable> Intersection<T> {
    pub fn new(t: f64, object: T) -> Self {
        Self { t, object }
    }

    pub fn hit(intersections: &mut [Intersection<T>]) -> Option<&Intersection<T>> {
        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
        intersections
            .iter()
            .find(|intersection| intersection.t >= 0.0)
    }
}

#[derive(Debug, Clone)]
pub struct Sphere {
    pub id: Uuid,
    pub radius: f64,
    pub center: Point3<f64>,
    pub transform: Transform,
}

impl Sphere {
    pub fn new(radius: f64, center: Point3<f64>, transform: Transform) -> Self {
        Sphere {
            id: Uuid::new_v4(),
            radius,
            center,
            transform,
        }
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Sphere {
            id: Uuid::new_v4(),
            radius: 1.0,
            center: Point3::new(0.0, 0.0, 0.0),
            transform: Transform::Identity,
        }
    }
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<Self>>, TransformError> {
        let ray2 = Ray::new(
            self.transform.apply_inverse(ray.origin)?,
            self.transform.apply_inverse(ray.direction)?,
        );
        let sphere_to_ray = ray2.origin - self.center;
        let a = ray2.direction.dot(&ray2.direction);
        let b = 2.0 * ray2.direction.dot(&sphere_to_ray);
        let c = sphere_to_ray.dot(&sphere_to_ray) - 1.0;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return Ok(vec![]);
        }
        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
        Ok(vec![
            Intersection::new(t1, self.clone()),
            Intersection::new(t2, self.clone()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::Axis;
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
    fn test_ray_intersects_sphere_at_two_points() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert_abs_diff_eq!(intersections[0].t, 4.0);
        assert_abs_diff_eq!(intersections[1].t, 6.0);
    }

    #[test]
    fn test_ray_intersects_sphere_at_a_tangent() {
        let ray = Ray::new(Point3::new(0.0, 1.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert_abs_diff_eq!(intersections[0].t, 5.0);
        assert_abs_diff_eq!(intersections[1].t, 5.0);
    }

    #[test]
    fn test_ray_misses_a_sphere() {
        let ray = Ray::new(Point3::new(0.0, 2.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersections = sphere.intersect(&ray).unwrap();

        assert!(intersections.is_empty());
    }

    #[test]
    fn test_ray_originates_inside_a_sphere() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert_abs_diff_eq!(intersections[0].t, -1.0);
        assert_abs_diff_eq!(intersections[1].t, 1.0);
    }

    #[test]
    fn test_sphere_is_behind_a_ray() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert_abs_diff_eq!(intersections[0].t, -6.0);
        assert_abs_diff_eq!(intersections[1].t, -4.0);
    }

    #[test]
    fn test_intersection_encapsulates_t_and_object() {
        let sphere = Sphere::default();
        let intersection = Intersection::new(3.5, sphere.clone());

        assert_abs_diff_eq!(intersection.t, 3.5);
        assert_eq!(intersection.object.id, sphere.id);
    }

    #[test]
    fn test_intersect_sets_the_object_on_the_intersection() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert_eq!(intersections[0].object.id, sphere.id);
        assert_eq!(intersections[1].object.id, sphere.id);
    }

    #[test]
    fn test_hit_when_all_intersections_have_positive_t() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(1.0, sphere.clone());
        let i2 = Intersection::new(2.0, sphere.clone());
        let mut intersections = vec![i1, i2];

        let hit = Intersection::hit(&mut intersections).unwrap();
        assert_abs_diff_eq!(hit.t, 1.0);
    }

    #[test]
    fn test_hit_when_some_intersections_have_negative_t() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(-1.0, sphere.clone());
        let i2 = Intersection::new(1.0, sphere.clone());
        let mut intersections = vec![i1, i2];

        let hit = Intersection::hit(&mut intersections).unwrap();
        assert_abs_diff_eq!(hit.t, 1.0);
    }

    #[test]
    fn test_hit_when_all_intersections_have_negative_t() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(-2.0, sphere.clone());
        let i2 = Intersection::new(-1.0, sphere.clone());
        let mut intersections = vec![i1, i2];

        let hit = Intersection::hit(&mut intersections);
        assert!(hit.is_none());
    }

    #[test]
    fn test_hit_is_lowest_nonnegative_intersection() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(5.0, sphere.clone());
        let i2 = Intersection::new(7.0, sphere.clone());
        let i3 = Intersection::new(-3.0, sphere.clone());
        let i4 = Intersection::new(2.0, sphere.clone());
        let mut intersections = vec![i1, i2, i3, i4];

        let hit = Intersection::hit(&mut intersections).unwrap();
        assert_abs_diff_eq!(hit.t, 2.0);
    }

    #[test]
    fn test_translating_a_ray() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 1.0, 0.0));
        let transform = Transform::Translate(Translation3::new(3.0, 4.0, 5.0));
        let transformed_ray = ray.transform(&transform);

        assert_eq!(transformed_ray.origin, Point3::new(4.0, 6.0, 8.0));
        assert_eq!(transformed_ray.direction, Vector3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_scaling_a_ray() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 1.0, 0.0));
        let transform = Transform::Scale(Scale3::new(2.0, 3.0, 4.0));
        let transformed_ray = ray.transform(&transform);

        assert_eq!(transformed_ray.origin, Point3::new(2.0, 6.0, 12.0));
        assert_eq!(transformed_ray.direction, Vector3::new(0.0, 3.0, 0.0));
    }

    #[test]
    fn test_rotating_a_ray() {
        let ray = Ray::new(Point3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        let transform = Transform::Rotate {
            axis: Axis::X,
            angle: std::f64::consts::FRAC_PI_2,
        };
        let transformed_ray = ray.transform(&transform);

        assert_abs_diff_eq!(transformed_ray.origin.x, 0.0);
        assert_abs_diff_eq!(transformed_ray.origin.y, 0.0);
        assert_abs_diff_eq!(transformed_ray.origin.z, 1.0);
        assert_abs_diff_eq!(transformed_ray.direction.x, 0.0);
        assert_abs_diff_eq!(transformed_ray.direction.y, 0.0);
        assert_abs_diff_eq!(transformed_ray.direction.z, 1.0);
    }

    #[test]
    fn test_sphere_default_transformation() {
        let sphere = Sphere::default();

        assert_eq!(sphere.transform, Transform::Identity);
    }

    #[test]
    fn test_changing_sphere_transformation() {
        let sphere = Sphere {
            transform: Transform::Translate(Translation3::new(2.0, 3.0, 4.0)),
            ..Default::default()
        };

        assert_eq!(
            sphere.transform,
            Transform::Translate(Translation3::new(2.0, 3.0, 4.0))
        );
    }

    #[test]
    fn test_intersecting_scaled_sphere_with_ray() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere {
            transform: Transform::Scale(Scale3::new(2.0, 2.0, 2.0)),
            ..Default::default()
        };
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert_abs_diff_eq!(intersections[0].t, 3.0);
        assert_abs_diff_eq!(intersections[1].t, 7.0);
    }

    #[test]
    fn test_intersecting_translated_sphere_with_ray() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere {
            transform: Transform::Translate(Translation3::new(5.0, 0.0, 0.0)),
            ..Default::default()
        };
        let intersections = sphere.intersect(&ray).unwrap();

        assert!(intersections.is_empty());
    }
}
