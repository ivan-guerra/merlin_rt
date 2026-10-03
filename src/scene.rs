use crate::{
    canvas::Color,
    light::PointLight,
    material::Material,
    ray::Ray,
    shape::{Intersection, Shape},
    sphere::Sphere,
    transforms::{Transform, TransformError},
};

use nalgebra::{Matrix4, Point3, Scale3, Translation3, Vector3};

#[derive(Debug)]
pub struct World {
    light: PointLight,
    objects: Vec<Box<dyn Shape>>,
}

impl World {
    pub fn view_transform(from: Point3<f64>, to: Point3<f64>, up: Vector3<f64>) -> Transform {
        let forward = (to - from).normalize();
        let upn = up.normalize();
        let left = forward.cross(&upn);
        let true_up = left.cross(&forward);

        let orientation = Matrix4::new(
            left.x, left.y, left.z, 0.0, true_up.x, true_up.y, true_up.z, 0.0, -forward.x,
            -forward.y, -forward.z, 0.0, 0.0, 0.0, 0.0, 1.0,
        );

        Transform::sequence([
            Transform::translation(Translation3::new(-from.x, -from.y, -from.z)),
            Transform::from_matrix(orientation),
        ])
    }

    pub fn new(light: PointLight, objects: Vec<Box<dyn Shape>>) -> Self {
        World { light, objects }
    }

    pub fn intersect<'a>(&'a self, ray: &Ray) -> Vec<Intersection<'a>> {
        let mut intersections = Vec::new();

        for object in &self.objects {
            if let Ok(mut object_intersections) = object.intersect(ray) {
                intersections.append(&mut object_intersections);
            }
        }

        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
        intersections
    }

    pub fn shade_hit(&self, comps: &Computations) -> Result<Color, TransformError> {
        let shadowed = self.is_shadowed(comps.over_point);
        comps
            .object
            .lighting(self.light, comps.point, comps.eyev, comps.normalv, shadowed)
    }

    pub fn color_at(&self, ray: &Ray) -> Result<Color, TransformError> {
        let mut intersections = self.intersect(ray);
        if let Some(hit) = Intersection::hit(&mut intersections) {
            let comps = Computations::prepare_computations(hit, ray).unwrap();
            Ok(self.shade_hit(&comps)?)
        } else {
            Ok(Color::new(0.0, 0.0, 0.0))
        }
    }

    pub fn is_shadowed(&self, point: Point3<f64>) -> bool {
        let v = self.light.position - point;
        let distance = v.magnitude();
        let direction = v.normalize();

        let r = Ray::new(point, direction);
        let mut intersections = self.intersect(&r);

        if let Some(hit) = Intersection::hit(&mut intersections) {
            hit.t < distance
        } else {
            false
        }
    }
}

impl Default for World {
    fn default() -> Self {
        let light = PointLight::new(
            Point3::new(-10.0, 10.0, -10.0),
            crate::canvas::Color::new(1.0, 1.0, 1.0),
        );

        let sphere1 = Sphere {
            material: Material {
                color: crate::canvas::Color::new(0.8, 1.0, 0.6),
                diffuse: 0.7,
                specular: 0.2,
                ..Default::default()
            },
            ..Default::default()
        };
        let sphere2 = Sphere {
            transform: Transform::scale(Scale3::new(0.5, 0.5, 0.5)),
            ..Default::default()
        };

        World {
            light,
            objects: vec![Box::new(sphere1), Box::new(sphere2)],
        }
    }
}

#[derive(Debug)]
pub struct Computations<'a> {
    pub t: f64,
    pub object: &'a dyn Shape,
    pub point: Point3<f64>,
    pub over_point: Point3<f64>,
    pub eyev: Vector3<f64>,
    pub normalv: Vector3<f64>,
    pub inside: bool,
}

impl Computations<'_> {
    pub fn prepare_computations<'a>(
        intersection: &Intersection<'a>,
        ray: &Ray,
    ) -> Result<Computations<'a>, TransformError> {
        let t = intersection.t;
        let object = intersection.object;
        let point = ray.position(t);
        let eyev = -ray.direction;
        let mut normalv = object.normal_at(point)?;
        let inside = normalv.dot(&eyev) < 0.0;

        if inside {
            normalv = -normalv;
        }

        const EPSILON: f64 = 1e-5;
        let over_point = point + normalv * EPSILON;

        Ok(Computations {
            t,
            object,
            point,
            over_point,
            eyev,
            normalv,
            inside,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_default_world() {
        let world = World::default();

        assert_eq!(world.light.position, Point3::new(-10.0, 10.0, -10.0));
        assert_eq!(
            world.light.intensity,
            crate::canvas::Color::new(1.0, 1.0, 1.0)
        );
        assert_eq!(world.objects.len(), 2);
    }

    #[test]
    fn test_intersect_a_world_with_a_ray() {
        let world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));

        let intersections = world.intersect(&ray);

        assert_eq!(intersections.len(), 4);
        assert_abs_diff_eq!(intersections[0].t, 4.0);
        assert_abs_diff_eq!(intersections[1].t, 4.5);
        assert_abs_diff_eq!(intersections[2].t, 5.5);
        assert_abs_diff_eq!(intersections[3].t, 6.0);
    }

    #[test]
    fn test_precomputing_the_state_of_an_intersection() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersection = Intersection::new(4.0, &sphere);
        let comps = Computations::prepare_computations(&intersection, &ray).unwrap();

        assert_abs_diff_eq!(comps.t, intersection.t);
        assert!(std::ptr::eq(comps.object, intersection.object));
        assert_abs_diff_eq!(comps.point, Point3::new(0.0, 0.0, -1.0));
        assert_abs_diff_eq!(comps.eyev, Vector3::new(0.0, 0.0, -1.0));
        assert_abs_diff_eq!(comps.normalv, Vector3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn test_the_hit_when_an_intersection_occurs_on_the_outside() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersection = Intersection::new(4.0, &sphere);
        let comps = Computations::prepare_computations(&intersection, &ray).unwrap();

        assert!(!comps.inside);
    }

    #[test]
    fn test_the_hit_when_an_intersection_occurs_on_the_inside() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersection = Intersection::new(1.0, &sphere);
        let comps = Computations::prepare_computations(&intersection, &ray).unwrap();

        assert_abs_diff_eq!(comps.point, Point3::new(0.0, 0.0, 1.0));
        assert_abs_diff_eq!(comps.eyev, Vector3::new(0.0, 0.0, -1.0));
        assert!(comps.inside);
        assert_abs_diff_eq!(comps.normalv, Vector3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn test_shading_an_intersection() {
        let world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = world.objects[0].as_ref();
        let intersection = Intersection::new(4.0, sphere);
        let comps = Computations::prepare_computations(&intersection, &ray).unwrap();
        let color = world.shade_hit(&comps).unwrap();

        assert_eq!(color, Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn test_shading_an_intersection_from_the_inside() {
        let world = World {
            light: PointLight::new(
                Point3::new(0.0, 0.25, 0.0),
                crate::canvas::Color::new(1.0, 1.0, 1.0),
            ),
            ..Default::default()
        };
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = world.objects[1].as_ref();
        let intersection = Intersection::new(0.5, sphere);
        let comps = Computations::prepare_computations(&intersection, &ray).unwrap();
        let color = world.shade_hit(&comps).unwrap();

        assert_eq!(color, Color::new(0.90498, 0.90498, 0.90498));
    }

    #[test]
    fn test_color_when_a_ray_misses() {
        let world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 1.0, 0.0));
        let color = world.color_at(&ray).unwrap();

        assert_eq!(color, Color::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_color_when_a_ray_hits() {
        let world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let color = world.color_at(&ray).unwrap();

        assert_eq!(color, Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn test_color_with_an_intersection_behind_the_ray() {
        let mut world = World::default();
        world.objects[0].material_mut().ambient = 1.0;
        world.objects[1].material_mut().ambient = 1.0;
        let expected_color = world.objects[1].material().color;

        let ray = Ray::new(Point3::new(0.0, 0.0, 0.75), Vector3::new(0.0, 0.0, -1.0));
        let color = world.color_at(&ray).unwrap();

        assert_eq!(color, expected_color);
    }

    #[test]
    fn test_transformation_matrix_for_the_default_orientation() {
        let from = Point3::new(0.0, 0.0, 0.0);
        let to = Point3::new(0.0, 0.0, -1.0);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let t = World::view_transform(from, to, up);

        assert_eq!(t, Transform::identity());
    }

    #[test]
    fn test_view_transformation_matrix_looking_in_positive_z_direction() {
        let from = Point3::new(0.0, 0.0, 0.0);
        let to = Point3::new(0.0, 0.0, 1.0);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let t = World::view_transform(from, to, up);

        assert_eq!(t, Transform::scale(Scale3::new(-1.0, 1.0, -1.0)));
    }

    #[test]
    fn test_view_transformation_moves_the_world() {
        let from = Point3::new(0.0, 0.0, 8.0);
        let to = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let t = World::view_transform(from, to, up);

        assert_eq!(t, Transform::translation(Translation3::new(0.0, 0.0, -8.0)));
    }

    #[test]
    fn test_an_arbitrary_view_transformation() {
        let from = Point3::new(1.0, 3.0, 2.0);
        let to = Point3::new(4.0, -2.0, 8.0);
        let up = Vector3::new(1.0, 1.0, 0.0);
        let t = World::view_transform(from, to, up);

        let expected = Matrix4::new(
            -0.50709, 0.50709, 0.67612, -2.36643, 0.76772, 0.60609, 0.12122, -2.82843, -0.35857,
            0.59761, -0.71714, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000,
        );

        assert_abs_diff_eq!(*t.matrix(), expected, epsilon = 1e-5);
    }

    #[test]
    fn test_there_is_no_shadow_when_nothing_is_collinear_with_point_and_light() {
        let world = World::default();
        let point = Point3::new(0.0, 10.0, 0.0);

        assert!(!world.is_shadowed(point));
    }

    #[test]
    fn test_the_shadow_when_an_object_is_between_the_point_and_the_light() {
        let world = World::default();
        let point = Point3::new(10.0, -10.0, 10.0);

        assert!(world.is_shadowed(point));
    }

    #[test]
    fn test_there_is_no_shadow_when_an_object_is_behind_the_light() {
        let world = World::default();
        let point = Point3::new(-20.0, 20.0, -20.0);

        assert!(!world.is_shadowed(point));
    }

    #[test]
    fn test_there_is_no_shadow_when_an_object_is_behind_the_point() {
        let world = World::default();
        let point = Point3::new(-2.0, 2.0, -2.0);

        assert!(!world.is_shadowed(point));
    }

    #[test]
    fn test_shade_hit_is_given_an_intersection_in_shadow() {
        let mut world = World::default();
        let light = PointLight::new(
            Point3::new(0.0, 0.0, -10.0),
            crate::canvas::Color::new(1.0, 1.0, 1.0),
        );
        world.light = light;
        let sphere2 = world.objects[1].as_ref();
        let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, 1.0));
        let intersection = Intersection::new(4.0, sphere2);
        let comps = Computations::prepare_computations(&intersection, &ray).unwrap();
        let color = world.shade_hit(&comps).unwrap();

        assert_eq!(color, Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn test_the_hit_should_offset_the_point() {
        let mut world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere {
            transform: Transform::translation(Translation3::new(0.0, 0.0, 1.0)),
            ..Default::default()
        };
        world.objects[0] = Box::new(sphere);
        let intersection = Intersection::new(5.0, world.objects[0].as_ref());
        let comps = Computations::prepare_computations(&intersection, &ray).unwrap();

        assert!(comps.over_point.z < -f64::EPSILON / 2.0);
        assert!(comps.point.z > comps.over_point.z);
    }
}
