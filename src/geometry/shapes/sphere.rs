use crate::{
    geometry::{
        ray::Ray,
        shapes::{Intersection, ParentLink, Shape},
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};

use nalgebra::{Point3, Vector3};
use uuid::Uuid;

#[derive(Debug, PartialEq)]
pub struct Sphere {
    parent: ParentLink,
    id: Uuid,
    radius: f64,
    center: Point3<f64>,
    transform: Transform,
    material: Material,
}

impl Sphere {
    pub fn builder() -> SphereBuilder {
        SphereBuilder::default()
    }

    pub fn new(radius: f64, center: Point3<f64>, transform: Transform, material: Material) -> Self {
        Self::builder()
            .radius(radius)
            .center(center)
            .transform(transform)
            .material(material)
            .build()
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Sphere {
            id: Uuid::new_v4(),
            radius: 1.0,
            center: Point3::new(0.0, 0.0, 0.0),
            parent: ParentLink::default(),
            transform: Transform::identity(),
            material: Material::default(),
        }
    }
}

#[derive(Debug, Default)]
#[must_use = "call build() to create the shape"]
pub struct SphereBuilder {
    shape: Sphere,
}

impl SphereBuilder {
    pub fn radius(mut self, radius: f64) -> Self {
        self.shape.radius = radius;
        self
    }

    pub fn center(mut self, center: Point3<f64>) -> Self {
        self.shape.center = center;
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

    pub fn build(self) -> Sphere {
        self.shape
    }
}

impl Shape for Sphere {
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
        let sphere_to_ray = ray.origin - self.center;
        let a = ray.direction.dot(&ray.direction);
        let b = 2.0 * ray.direction.dot(&sphere_to_ray);
        let c = sphere_to_ray.dot(&sphere_to_ray) - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return Ok(vec![]);
        }

        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
        let t2 = (-b + discriminant.sqrt()) / (2.0 * a);

        Ok(vec![
            Intersection::new(t1, self),
            Intersection::new(t2, self),
        ])
    }

    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        let object_point = self.world_to_object(world_point)?;
        let object_normal = (object_point - self.center) / self.radius;
        self.normal_to_world(object_normal)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::EPSILON;
    use crate::{
        rendering::canvas::Color,
        scene::{light::PointLight, material::Material},
    };
    use approx::assert_abs_diff_eq;
    use nalgebra::{Point3, Scale3, Translation3, Vector3};

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
        let intersection = Intersection::new(3.5, &sphere);

        assert_abs_diff_eq!(intersection.t, 3.5);
        assert!(std::ptr::eq(intersection.object, &sphere as &dyn Shape));
    }

    #[test]
    fn test_intersect_sets_the_object_on_the_intersection() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert!(std::ptr::addr_eq(
            intersections[0].object,
            &sphere as &dyn Shape
        ));
        assert!(std::ptr::addr_eq(
            intersections[1].object,
            &sphere as &dyn Shape
        ));
    }

    #[test]
    fn test_hit_when_all_intersections_have_positive_t() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(1.0, &sphere);
        let i2 = Intersection::new(2.0, &sphere);
        let mut intersections = vec![i1, i2];

        let hit = Intersection::hit(&mut intersections).unwrap();
        assert_abs_diff_eq!(hit.t, 1.0);
    }

    #[test]
    fn test_hit_when_some_intersections_have_negative_t() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(-1.0, &sphere);
        let i2 = Intersection::new(1.0, &sphere);
        let mut intersections = vec![i1, i2];

        let hit = Intersection::hit(&mut intersections).unwrap();
        assert_abs_diff_eq!(hit.t, 1.0);
    }

    #[test]
    fn test_hit_when_all_intersections_have_negative_t() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(-2.0, &sphere);
        let i2 = Intersection::new(-1.0, &sphere);
        let mut intersections = vec![i1, i2];

        let hit = Intersection::hit(&mut intersections);
        assert!(hit.is_none());
    }

    #[test]
    fn test_hit_is_lowest_nonnegative_intersection() {
        let sphere = Sphere::default();
        let i1 = Intersection::new(5.0, &sphere);
        let i2 = Intersection::new(7.0, &sphere);
        let i3 = Intersection::new(-3.0, &sphere);
        let i4 = Intersection::new(2.0, &sphere);
        let mut intersections = vec![i1, i2, i3, i4];

        let hit = Intersection::hit(&mut intersections).unwrap();
        assert_abs_diff_eq!(hit.t, 2.0);
    }

    #[test]
    fn test_sphere_default_transformation() {
        let sphere = Sphere::default();

        assert_eq!(sphere.transform, Transform::identity());
    }

    #[test]
    fn test_changing_sphere_transformation() {
        let sphere = Sphere::builder()
            .transform(Transform::translation(Translation3::new(2.0, 3.0, 4.0)))
            .build();

        assert_eq!(
            sphere.transform,
            Transform::translation(Translation3::new(2.0, 3.0, 4.0))
        );
    }

    #[test]
    fn test_intersecting_scaled_sphere_with_ray() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::builder()
            .transform(Transform::scale(Scale3::new(2.0, 2.0, 2.0)))
            .build();
        let intersections = sphere.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 2);
        assert_abs_diff_eq!(intersections[0].t, 3.0);
        assert_abs_diff_eq!(intersections[1].t, 7.0);
    }

    #[test]
    fn test_intersecting_translated_sphere_with_ray() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::builder()
            .transform(Transform::translation(Translation3::new(5.0, 0.0, 0.0)))
            .build();
        let intersections = sphere.intersect(&ray).unwrap();

        assert!(intersections.is_empty());
    }

    #[test]
    fn test_normal_on_sphere_at_point_on_x_axis() {
        let sphere = Sphere::default();
        let normal = sphere.normal_at(Point3::new(1.0, 0.0, 0.0)).unwrap();

        assert_abs_diff_eq!(normal, Vector3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_normal_on_sphere_at_point_on_y_axis() {
        let sphere = Sphere::default();
        let normal = sphere.normal_at(Point3::new(0.0, 1.0, 0.0)).unwrap();

        assert_abs_diff_eq!(normal, Vector3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_normal_on_sphere_at_point_on_z_axis() {
        let sphere = Sphere::default();
        let normal = sphere.normal_at(Point3::new(0.0, 0.0, 1.0)).unwrap();

        assert_abs_diff_eq!(normal, Vector3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_normal_on_sphere_at_nonaxial_point() {
        let sqrt_3_over_3: f64 = 3_f64.sqrt() / 3.0;
        let sphere = Sphere::default();
        let normal = sphere
            .normal_at(Point3::new(sqrt_3_over_3, sqrt_3_over_3, sqrt_3_over_3))
            .unwrap();

        assert_abs_diff_eq!(
            normal,
            Vector3::new(sqrt_3_over_3, sqrt_3_over_3, sqrt_3_over_3)
        );
    }

    #[test]
    fn test_normal_is_normalized_vector() {
        let sqrt_3_over_3: f64 = 3_f64.sqrt() / 3.0;
        let sphere = Sphere::default();
        let normal = sphere
            .normal_at(Point3::new(sqrt_3_over_3, sqrt_3_over_3, sqrt_3_over_3))
            .unwrap();

        assert_abs_diff_eq!(normal, normal.normalize());
    }

    #[test]
    fn test_computing_normal_on_translated_sphere() {
        let sphere = Sphere::builder()
            .transform(Transform::translation(Translation3::new(0.0, 1.0, 0.0)))
            .build();
        let normal = sphere
            .normal_at(Point3::new(0.0, 1.70711, -std::f64::consts::FRAC_1_SQRT_2))
            .unwrap();

        assert_abs_diff_eq!(
            normal,
            Vector3::new(0.0, 0.707117, -std::f64::consts::FRAC_1_SQRT_2),
            epsilon = EPSILON
        );
    }

    #[test]
    fn test_computing_normal_on_transformed_sphere() {
        let sphere = Sphere::builder()
            .transform(Transform::sequence([
                Transform::rotation(
                    crate::geometry::transforms::Axis::Z,
                    std::f64::consts::PI / 5.0,
                ),
                Transform::scale(Scale3::new(1.0, 0.5, 1.0)),
            ]))
            .build();
        let normal = sphere
            .normal_at(Point3::new(0.0, 2f64.sqrt() / 2.0, -2f64.sqrt() / 2.0))
            .unwrap();

        assert_abs_diff_eq!(
            normal,
            Vector3::new(0.0, 0.97014, -0.24254),
            epsilon = EPSILON
        );
    }

    #[test]
    fn test_sphere_has_a_default_material() {
        let sphere = Sphere::default();
        let default_material = Material::default();

        assert_eq!(sphere.material.color, default_material.color);
        assert_abs_diff_eq!(sphere.material.ambient, default_material.ambient);
        assert_abs_diff_eq!(sphere.material.diffuse, default_material.diffuse);
        assert_abs_diff_eq!(sphere.material.specular, default_material.specular);
        assert_abs_diff_eq!(sphere.material.shininess, default_material.shininess);
    }

    #[test]
    fn test_sphere_can_be_assigned_a_material() {
        let material = Material {
            ambient: 1.0,
            ..Default::default()
        };
        let sphere = Sphere::builder().material(material).build();

        assert_abs_diff_eq!(sphere.material.ambient, 1.0);
    }

    #[test]
    fn test_lighting_with_eye_between_the_light_and_the_surface() {
        let sphere = Sphere::default();
        let position = Point3::new(0.0, 0.0, 0.0);
        let eyev = Vector3::new(0.0, 0.0, -1.0);
        let normalv = Vector3::new(0.0, 0.0, -1.0);
        let light = PointLight::new(Point3::new(0.0, 0.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let result = sphere
            .lighting(light, position, eyev, normalv, false)
            .unwrap();

        assert_abs_diff_eq!(result.r(), 1.9);
        assert_abs_diff_eq!(result.g(), 1.9);
        assert_abs_diff_eq!(result.b(), 1.9);
    }

    #[test]
    fn test_lighting_with_eye_between_light_and_surface_eye_offset_45_degrees() {
        let sphere = Sphere::default();
        let position = Point3::new(0.0, 0.0, 0.0);
        let eyev = Vector3::new(0.0, 2f64.sqrt() / 2.0, -2f64.sqrt() / 2.0);
        let normalv = Vector3::new(0.0, 0.0, -1.0);
        let light = PointLight::new(Point3::new(0.0, 0.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let result = sphere
            .lighting(light, position, eyev, normalv, false)
            .unwrap();

        assert_abs_diff_eq!(result.r(), 1.0);
        assert_abs_diff_eq!(result.g(), 1.0);
        assert_abs_diff_eq!(result.b(), 1.0);
    }

    #[test]
    fn test_lighting_with_eye_opposite_surface_light_offset_45_degrees() {
        let sphere = Sphere::default();
        let position = Point3::new(0.0, 0.0, 0.0);
        let eyev = Vector3::new(0.0, 0.0, -1.0);
        let normalv = Vector3::new(0.0, 0.0, -1.0);
        let light = PointLight::new(Point3::new(0.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let result = sphere
            .lighting(light, position, eyev, normalv, false)
            .unwrap();

        assert_abs_diff_eq!(result.r(), 0.736396, epsilon = EPSILON);
        assert_abs_diff_eq!(result.g(), 0.736396, epsilon = EPSILON);
        assert_abs_diff_eq!(result.b(), 0.736396, epsilon = EPSILON);
    }

    #[test]
    fn test_lighting_with_eye_in_path_of_reflection_vector() {
        let sphere = Sphere::default();
        let position = Point3::new(0.0, 0.0, 0.0);
        let eyev = Vector3::new(0.0, -2f64.sqrt() / 2.0, -2f64.sqrt() / 2.0);
        let normalv = Vector3::new(0.0, 0.0, -1.0);
        let light = PointLight::new(Point3::new(0.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let result = sphere
            .lighting(light, position, eyev, normalv, false)
            .unwrap();

        assert_abs_diff_eq!(result.r(), 1.636396, epsilon = EPSILON);
        assert_abs_diff_eq!(result.g(), 1.636396, epsilon = EPSILON);
        assert_abs_diff_eq!(result.b(), 1.636396, epsilon = EPSILON);
    }

    #[test]
    fn test_lighting_with_light_behind_surface() {
        let sphere = Sphere::default();
        let position = Point3::new(0.0, 0.0, 0.0);
        let eyev = Vector3::new(0.0, 0.0, -1.0);
        let normalv = Vector3::new(0.0, 0.0, -1.0);
        let light = PointLight::new(Point3::new(0.0, 0.0, 10.0), Color::new(1.0, 1.0, 1.0));
        let result = sphere
            .lighting(light, position, eyev, normalv, false)
            .unwrap();

        assert_abs_diff_eq!(result.r(), 0.1);
        assert_abs_diff_eq!(result.g(), 0.1);
        assert_abs_diff_eq!(result.b(), 0.1);
    }

    #[test]
    fn test_lighting_with_surface_in_shadow() {
        let sphere = Sphere::default();
        let position = Point3::new(0.0, 0.0, 0.0);
        let eyev = Vector3::new(0.0, 0.0, -1.0);
        let normalv = Vector3::new(0.0, 0.0, -1.0);
        let light = PointLight::new(Point3::new(0.0, 0.0, -10.0), Color::new(1.0, 1.0, 1.0));
        let in_shadow = true;
        let result = sphere
            .lighting(light, position, eyev, normalv, in_shadow)
            .unwrap();

        assert_abs_diff_eq!(result.r(), 0.1);
        assert_abs_diff_eq!(result.g(), 0.1);
        assert_abs_diff_eq!(result.b(), 0.1);
    }
}
