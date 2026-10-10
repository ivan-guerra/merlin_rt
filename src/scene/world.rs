use crate::{
    geometry::{
        ray::Ray,
        shapes::{Intersection, Plane, Shape, ShapeRef, Sphere},
        transforms::{Transform, TransformError},
    },
    rendering::canvas::Color,
    scene::{light::PointLight, material::Material},
};

use nalgebra::{Matrix4, Point3, Scale3, Translation3, Vector3};
use std::rc::Rc;

pub const MAX_RECURSION_DEPTH: usize = 5;

#[derive(Debug)]
pub struct World {
    light: PointLight,
    objects: Vec<ShapeRef>,
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
        Self::from_shared(light, objects.into_iter().map(Rc::from).collect())
    }

    /// Creates a world from shared root shapes, including groups.
    /// Do not also insert a group's descendants as world roots.
    pub fn from_shared(light: PointLight, objects: Vec<ShapeRef>) -> Self {
        World { light, objects }
    }

    pub fn intersect<'a>(&'a self, ray: &Ray) -> Result<Vec<Intersection<'a>>, TransformError> {
        let mut intersections = Vec::new();

        for object in &self.objects {
            let mut object_intersections = object.intersect(ray)?;
            intersections.append(&mut object_intersections);
        }

        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
        Ok(intersections)
    }

    pub fn shade_hit(
        &self,
        comps: &Computations,
        remaining: usize,
    ) -> Result<Color, TransformError> {
        let shadowed = self.is_shadowed(comps.over_point);
        let surface = comps.object.lighting(
            self.light,
            comps.point,
            comps.eyev,
            comps.normalv,
            shadowed.unwrap(),
        )?;
        let reflected = self.reflected_color(comps, remaining)?;
        let refracted = self.refracted_color(comps, remaining)?;

        if comps.object.material().reflective > 0.0 && comps.object.material().transparency > 0.0 {
            let reflectance = comps.schlick();
            Ok(surface + reflected * reflectance + refracted * (1.0 - reflectance))
        } else {
            Ok(surface + reflected + refracted)
        }
    }

    pub fn color_at(&self, ray: &Ray, remaining: usize) -> Result<Color, TransformError> {
        let intersections = self.intersect(ray)?;

        if let Some(hit_index) = intersections
            .iter()
            .position(|intersection| intersection.t >= 0.0)
        {
            let comps = Computations::prepare_computations(
                &intersections[hit_index],
                ray,
                Some(&intersections),
            )?;
            self.shade_hit(&comps, remaining)
        } else {
            Ok(Color::new(0.0, 0.0, 0.0))
        }
    }

    pub fn is_shadowed(&self, point: Point3<f64>) -> Result<bool, TransformError> {
        let v = self.light.position - point;
        let distance = v.magnitude();
        let direction = v.normalize();

        let r = Ray::new(point, direction);
        let mut intersections = self.intersect(&r)?;

        if let Some(hit) = Intersection::hit(&mut intersections) {
            Ok(hit.t < distance)
        } else {
            Ok(false)
        }
    }

    pub fn reflected_color(
        &self,
        comps: &Computations,
        remaining: usize,
    ) -> Result<Color, TransformError> {
        if remaining == 0 || comps.object.material().reflective == 0.0 {
            return Ok(Color::new(0.0, 0.0, 0.0));
        }

        let reflect_ray = Ray::new(comps.over_point, comps.reflectv);
        let color = self.color_at(&reflect_ray, remaining - 1)?;

        Ok(color * comps.object.material().reflective)
    }

    pub fn refracted_color(
        &self,
        comps: &Computations,
        remaining: usize,
    ) -> Result<Color, TransformError> {
        if remaining == 0 || comps.object.material().transparency == 0.0 {
            return Ok(Color::new(0.0, 0.0, 0.0));
        }

        let n_ratio = comps.n1.unwrap() / comps.n2.unwrap();
        let cos_i = comps.eyev.dot(&comps.normalv);
        let sin2_t = n_ratio * n_ratio * (1.0 - cos_i * cos_i);

        if sin2_t > 1.0 {
            return Ok(Color::new(0.0, 0.0, 0.0));
        }

        let cos_t = (1.0 - sin2_t).sqrt();
        let direction = comps.normalv * (n_ratio * cos_i - cos_t) - comps.eyev * n_ratio;
        let refract_ray = Ray::new(comps.under_point, direction);
        let color = self.color_at(&refract_ray, remaining - 1)?;

        Ok(color * comps.object.material().transparency)
    }
}

impl Default for World {
    fn default() -> Self {
        let light = PointLight::new(
            Point3::new(-10.0, 10.0, -10.0),
            crate::rendering::canvas::Color::new(1.0, 1.0, 1.0),
        );

        let sphere1 = Sphere::builder()
            .material(Material {
                color: crate::rendering::canvas::Color::new(0.8, 1.0, 0.6),
                diffuse: 0.7,
                specular: 0.2,
                ..Default::default()
            })
            .build();
        let sphere2 = Sphere::builder()
            .transform(Transform::scale(Scale3::new(0.5, 0.5, 0.5)))
            .build();
        let plane = Plane::builder()
            .material(Material {
                reflective: 0.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -1.0, 0.0)))
            .build();

        World {
            light,
            objects: vec![Rc::new(sphere1), Rc::new(sphere2), Rc::new(plane)],
        }
    }
}

#[derive(Debug)]
pub struct Computations<'a> {
    pub t: f64,
    pub object: &'a dyn Shape,
    pub point: Point3<f64>,
    pub over_point: Point3<f64>,
    pub under_point: Point3<f64>,
    pub eyev: Vector3<f64>,
    pub normalv: Vector3<f64>,
    pub reflectv: Vector3<f64>,
    pub inside: bool,
    pub n1: Option<f64>,
    pub n2: Option<f64>,
}

impl Computations<'_> {
    pub fn prepare_computations<'a>(
        intersection: &Intersection<'a>,
        ray: &Ray,
        xs: Option<&[Intersection<'a>]>,
    ) -> Result<Computations<'a>, TransformError> {
        let containers = if let Some(xs) = xs {
            let mut containers: Vec<&dyn Shape> = Vec::new();
            let mut n1 = None;
            let mut n2 = None;

            for i in xs {
                if std::ptr::eq(i, intersection) {
                    n1 = Some(
                        containers
                            .last()
                            .map_or(1.0, |shape| shape.material().refractive_index),
                    );
                }

                if let Some(pos) = containers
                    .iter()
                    .position(|&s| std::ptr::addr_eq(s, i.object))
                {
                    containers.remove(pos);
                } else {
                    containers.push(i.object);
                }

                if std::ptr::eq(i, intersection) {
                    n2 = Some(
                        containers
                            .last()
                            .map_or(1.0, |shape| shape.material().refractive_index),
                    );
                    break;
                }
            }

            (n1, n2)
        } else {
            (None, None)
        };

        let t = intersection.t;
        let object = intersection.object;
        let point = ray.position(t);
        let eyev = -ray.direction;
        let mut normalv = object.normal_at_hit(point, intersection)?;
        let inside = normalv.dot(&eyev) < 0.0;
        if inside {
            normalv = -normalv;
        }
        let reflectv = Transform::reflection(normalv).apply(ray.direction);

        const EPSILON: f64 = 1e-5;
        let over_point = point + normalv * EPSILON;
        let under_point = point - normalv * EPSILON;

        Ok(Computations {
            t,
            object,
            point,
            over_point,
            under_point,
            eyev,
            normalv,
            reflectv,
            inside,
            n1: containers.0,
            n2: containers.1,
        })
    }

    pub fn schlick(&self) -> f64 {
        let mut cos = self.eyev.dot(&self.normalv);

        if let (Some(n1), Some(n2)) = (self.n1, self.n2) {
            if n1 > n2 {
                let n = n1 / n2;
                let sin2_t = n * n * (1.0 - cos * cos);
                if sin2_t > 1.0 {
                    return 1.0;
                }
                let cos_t = (1.0 - sin2_t).sqrt();
                cos = cos_t;
            }

            let r0 = ((n1 - n2) / (n1 + n2)).powi(2);
            r0 + (1.0 - r0) * (1.0 - cos).powi(5)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::shapes::SphereBuilder, rendering::canvas::Color};
    use approx::assert_abs_diff_eq;

    fn glass_sphere_builder(refractive_index: f64) -> SphereBuilder {
        Sphere::builder().material(Material {
            transparency: 1.0,
            refractive_index,
            ..Default::default()
        })
    }

    #[derive(Debug)]
    struct TestPattern;

    impl crate::scene::pattern::Pattern for TestPattern {
        fn pattern_at(&self, point: Point3<f64>) -> Color {
            Color::new(point.x, point.y, point.z)
        }

        fn pattern_at_object(
            &self,
            object_transform: &Transform,
            world_point: Point3<f64>,
        ) -> Result<Color, TransformError> {
            let object_point = object_transform.apply_inverse(world_point)?;
            Ok(self.pattern_at(object_point))
        }
    }

    #[test]
    fn test_default_world() {
        let world = World::default();

        assert_eq!(world.light.position, Point3::new(-10.0, 10.0, -10.0));
        assert_eq!(world.light.intensity, Color::new(1.0, 1.0, 1.0));
        assert_eq!(world.objects.len(), 3);
    }

    #[test]
    fn test_intersect_a_world_with_a_ray() {
        let world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));

        let intersections = world.intersect(&ray).unwrap();

        assert_eq!(intersections.len(), 4);
        assert_eq!(intersections[0].t, 4.0);
        assert_eq!(intersections[1].t, 4.5);
        assert_eq!(intersections[2].t, 5.5);
        assert_eq!(intersections[3].t, 6.0);
    }

    #[test]
    fn test_precomputing_the_state_of_an_intersection() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersection = Intersection::new(4.0, &sphere);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();

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
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();

        assert!(!comps.inside);
    }

    #[test]
    fn test_the_hit_when_an_intersection_occurs_on_the_inside() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::default();
        let intersection = Intersection::new(1.0, &sphere);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();

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
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();
        let color = world.shade_hit(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn test_shading_an_intersection_from_the_inside() {
        let world = World {
            light: PointLight::new(Point3::new(0.0, 0.25, 0.0), Color::new(1.0, 1.0, 1.0)),
            ..Default::default()
        };
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = world.objects[1].as_ref();
        let intersection = Intersection::new(0.5, sphere);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();
        let color = world.shade_hit(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.90498, 0.90498, 0.90498));
    }

    #[test]
    fn test_color_when_a_ray_misses() {
        let world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 1.0, 0.0));
        let color = world.color_at(&ray, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_color_when_a_ray_hits() {
        let world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let color = world.color_at(&ray, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.38066, 0.47583, 0.2855));
    }

    #[test]
    fn test_color_with_an_intersection_behind_the_ray() {
        let mut world = World::default();
        world.objects[0] = Rc::new(
            Sphere::builder()
                .material(Material {
                    color: Color::new(0.8, 1.0, 0.6),
                    ambient: 1.0,
                    diffuse: 0.7,
                    specular: 0.2,
                    ..Default::default()
                })
                .build(),
        );
        world.objects[1] = Rc::new(
            Sphere::builder()
                .transform(Transform::scale(Scale3::new(0.5, 0.5, 0.5)))
                .material(Material {
                    ambient: 1.0,
                    ..Default::default()
                })
                .build(),
        );
        let expected_color = world.objects[1].material().color;

        let ray = Ray::new(Point3::new(0.0, 0.0, 0.75), Vector3::new(0.0, 0.0, -1.0));
        let color = world.color_at(&ray, MAX_RECURSION_DEPTH).unwrap();

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

        assert!(!world.is_shadowed(point).unwrap());
    }

    #[test]
    fn test_the_shadow_when_an_object_is_between_the_point_and_the_light() {
        let world = World::default();
        let point = Point3::new(10.0, -10.0, 10.0);

        assert!(world.is_shadowed(point).unwrap());
    }

    #[test]
    fn test_there_is_no_shadow_when_an_object_is_behind_the_light() {
        let world = World::default();
        let point = Point3::new(-20.0, 20.0, -20.0);

        assert!(!world.is_shadowed(point).unwrap());
    }

    #[test]
    fn test_there_is_no_shadow_when_an_object_is_behind_the_point() {
        let world = World::default();
        let point = Point3::new(-2.0, 2.0, -2.0);

        assert!(!world.is_shadowed(point).unwrap());
    }

    #[test]
    fn test_shade_hit_is_given_an_intersection_in_shadow() {
        let mut world = World::default();
        let light = PointLight::new(Point3::new(0.0, 0.0, -10.0), Color::new(1.0, 1.0, 1.0));
        world.light = light;
        let sphere2 = world.objects[1].as_ref();
        let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, 1.0));
        let intersection = Intersection::new(4.0, sphere2);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();
        let color = world.shade_hit(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.1, 0.1, 0.1));
    }

    #[test]
    fn test_the_hit_should_offset_the_point() {
        let mut world = World::default();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let sphere = Sphere::builder()
            .transform(Transform::translation(Translation3::new(0.0, 0.0, 1.0)))
            .build();
        world.objects[0] = Rc::new(sphere);
        let intersection = Intersection::new(5.0, world.objects[0].as_ref());
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();

        assert!(comps.over_point.z < -f64::EPSILON / 2.0);
        assert!(comps.point.z > comps.over_point.z);
    }

    #[test]
    fn test_precomputing_the_reflection_vector() {
        let shape = Plane::default();
        let ray = Ray::new(
            Point3::new(0.0, 1.0, -1.0),
            Vector3::new(
                0.0,
                -std::f64::consts::SQRT_2 / 2.0,
                std::f64::consts::SQRT_2 / 2.0,
            ),
        );
        let intersection = Intersection::new(std::f64::consts::SQRT_2, &shape);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();

        assert_abs_diff_eq!(
            comps.reflectv,
            Vector3::new(
                0.0,
                std::f64::consts::SQRT_2 / 2.0,
                std::f64::consts::SQRT_2 / 2.0
            )
        );
    }

    #[test]
    fn test_the_reflected_color_for_a_nonreflective_material() {
        let world = World::default();
        let shape = world.objects[1].as_ref();
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let intersection = Intersection::new(1.0, shape);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();
        let color = world.reflected_color(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_the_reflected_color_for_a_reflective_material() {
        let mut world = World::default();
        let plane = Plane::builder()
            .material(Material {
                reflective: 0.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -1.0, 0.0)))
            .build();
        world.objects.push(Rc::new(plane));
        let shape = world.objects.last().unwrap().as_ref();
        let ray = Ray::new(
            Point3::new(0.0, 0.0, -3.0),
            Vector3::new(
                0.0,
                -std::f64::consts::SQRT_2 / 2.0,
                std::f64::consts::SQRT_2 / 2.0,
            ),
        );
        let intersection = Intersection::new(std::f64::consts::SQRT_2, shape);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();
        let color = world.reflected_color(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.19033, 0.23791, 0.14274));
    }

    #[test]
    fn test_shade_hit_with_a_reflective_material() {
        let mut world = World::default();
        let plane = Plane::builder()
            .material(Material {
                reflective: 0.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -1.0, 0.0)))
            .build();
        world.objects.push(Rc::new(plane));
        let shape = world.objects.last().unwrap().as_ref();
        let ray = Ray::new(
            Point3::new(0.0, 0.0, -3.0),
            Vector3::new(
                0.0,
                -std::f64::consts::SQRT_2 / 2.0,
                std::f64::consts::SQRT_2 / 2.0,
            ),
        );
        let intersection = Intersection::new(std::f64::consts::SQRT_2, shape);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();
        let color = world.shade_hit(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.87677, 0.92436, 0.82918));
    }

    #[test]
    fn test_the_reflected_color_at_the_maximum_recursive_depth() {
        let mut world = World::default();
        let plane = Plane::builder()
            .material(Material {
                reflective: 0.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -1.0, 0.0)))
            .build();
        world.objects.push(Rc::new(plane));
        let shape = world.objects.last().unwrap().as_ref();
        let ray = Ray::new(
            Point3::new(0.0, 0.0, -3.0),
            Vector3::new(
                0.0,
                -std::f64::consts::SQRT_2 / 2.0,
                std::f64::consts::SQRT_2 / 2.0,
            ),
        );
        let intersection = Intersection::new(std::f64::consts::SQRT_2, shape);
        let comps = Computations::prepare_computations(&intersection, &ray, None).unwrap();
        let color = world.reflected_color(&comps, 0).unwrap();

        assert_eq!(color, Color::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_finding_n1_and_n2_at_various_intersections() {
        let a = glass_sphere_builder(1.5)
            .transform(Transform::scale(Scale3::new(2.0, 2.0, 2.0)))
            .build();
        let b = glass_sphere_builder(2.0)
            .transform(Transform::translation(Translation3::new(0.0, 0.0, -0.25)))
            .build();
        let c = glass_sphere_builder(2.5)
            .transform(Transform::translation(Translation3::new(0.0, 0.0, 0.25)))
            .build();

        let ray = Ray::new(Point3::new(0.0, 0.0, -4.0), Vector3::new(0.0, 0.0, 1.0));
        let intersections = vec![
            Intersection::new(2.0, &a),
            Intersection::new(2.75, &b),
            Intersection::new(3.25, &c),
            Intersection::new(4.75, &b),
            Intersection::new(5.25, &c),
            Intersection::new(6.0, &a),
        ];

        let expected_n1_n2 = [
            (1.0, 1.5),
            (1.5, 2.0),
            (2.0, 2.5),
            (2.5, 2.5),
            (2.5, 1.5),
            (1.5, 1.0),
        ];

        for (i, intersection) in intersections.iter().enumerate() {
            let comps =
                Computations::prepare_computations(intersection, &ray, Some(&intersections))
                    .unwrap();
            assert_abs_diff_eq!(comps.n1.unwrap(), expected_n1_n2[i].0);
            assert_abs_diff_eq!(comps.n2.unwrap(), expected_n1_n2[i].1);
        }
    }

    #[test]
    fn test_finding_n1_and_n2_for_coincident_intersections() {
        let shape = glass_sphere_builder(1.5).build();
        let ray = Ray::new(Point3::new(0.0, 1.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let intersections = vec![
            Intersection::new(5.0, &shape),
            Intersection::new(5.0, &shape),
        ];

        let entering =
            Computations::prepare_computations(&intersections[0], &ray, Some(&intersections))
                .unwrap();
        let exiting =
            Computations::prepare_computations(&intersections[1], &ray, Some(&intersections))
                .unwrap();

        assert_abs_diff_eq!(entering.n1.unwrap(), 1.0);
        assert_abs_diff_eq!(entering.n2.unwrap(), 1.5);
        assert_abs_diff_eq!(exiting.n1.unwrap(), 1.5);
        assert_abs_diff_eq!(exiting.n2.unwrap(), 1.0);
    }

    #[test]
    fn test_the_under_point_is_offset_below_the_surface() {
        let shape = glass_sphere_builder(1.5)
            .transform(Transform::translation(Translation3::new(0.0, 0.0, 1.0)))
            .build();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let intersections = vec![Intersection::new(5.0, &shape)];
        let comps =
            Computations::prepare_computations(&intersections[0], &ray, Some(&intersections))
                .unwrap();

        assert!(comps.under_point.z > f64::EPSILON / 2.0);
        assert!(comps.point.z < comps.under_point.z);
    }

    #[test]
    fn test_the_refracted_color_with_an_opaque_surface() {
        let world = World::default();
        let shape = world.objects[0].as_ref();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let intersections = vec![Intersection::new(4.0, shape)];
        let comps =
            Computations::prepare_computations(&intersections[0], &ray, Some(&intersections))
                .unwrap();
        let color = world.refracted_color(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_the_refracted_color_at_the_maximum_recursive_depth() {
        let world = World::default();
        let shape = world.objects[0].as_ref();
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let intersections = vec![Intersection::new(4.0, shape)];
        let comps =
            Computations::prepare_computations(&intersections[0], &ray, Some(&intersections))
                .unwrap();
        let color = world.refracted_color(&comps, 0).unwrap();

        assert_eq!(color, Color::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_the_refracted_color_with_a_refracted_ray() {
        let mut world = World::default();
        world.objects[0] = Rc::new(
            Sphere::builder()
                .material(Material {
                    color: Color::new(0.8, 1.0, 0.6),
                    ambient: 1.0,
                    diffuse: 0.7,
                    specular: 0.2,
                    pattern: Some(Box::new(TestPattern)),
                    ..Default::default()
                })
                .build(),
        );
        world.objects[1] = Rc::new(
            glass_sphere_builder(1.5)
                .transform(Transform::scale(Scale3::new(0.5, 0.5, 0.5)))
                .build(),
        );

        let shape_a = world.objects[0].as_ref();
        let shape_b = world.objects[1].as_ref();
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.1), Vector3::new(0.0, 1.0, 0.0));
        let intersections = vec![
            Intersection::new(-0.9899, shape_a),
            Intersection::new(-0.4899, shape_b),
            Intersection::new(0.4899, shape_b),
            Intersection::new(0.9899, shape_a),
        ];
        let comps =
            Computations::prepare_computations(&intersections[2], &ray, Some(&intersections))
                .unwrap();
        let color = world.refracted_color(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.0, 0.99888, 0.04725));
    }

    #[test]
    fn test_shade_hit_with_a_transparent_material() {
        let mut world = World::default();
        let floor = Plane::builder()
            .material(Material {
                transparency: 0.5,
                refractive_index: 1.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -1.0, 0.0)))
            .build();
        world.objects.push(Rc::new(floor));
        let ball = Sphere::builder()
            .material(Material {
                color: Color::new(1.0, 0.0, 0.0),
                ambient: 0.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -3.5, -0.5)))
            .build();
        world.objects.push(Rc::new(ball));
        let ray = Ray::new(
            Point3::new(0.0, 0.0, -3.0),
            Vector3::new(
                0.0,
                -std::f64::consts::SQRT_2 / 2.0,
                std::f64::consts::SQRT_2 / 2.0,
            ),
        );
        let intersections = vec![Intersection::new(
            std::f64::consts::SQRT_2,
            world.objects[3].as_ref(),
        )];
        let comps =
            Computations::prepare_computations(&intersections[0], &ray, Some(&intersections))
                .unwrap();
        let color = world.shade_hit(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.93642, 0.68642, 0.68642));
    }

    #[test]
    fn test_the_schlick_approximation_under_total_internal_reflection() {
        let shape = glass_sphere_builder(1.5)
            .transform(Transform::scale(Scale3::new(2.0, 2.0, 2.0)))
            .build();
        let ray = Ray::new(
            Point3::new(0.0, 0.0, std::f64::consts::SQRT_2 / 2.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let intersections = vec![
            Intersection::new(-std::f64::consts::SQRT_2 / 2.0, &shape),
            Intersection::new(std::f64::consts::SQRT_2 / 2.0, &shape),
        ];
        let comps =
            Computations::prepare_computations(&intersections[1], &ray, Some(&intersections))
                .unwrap();
        let reflectance = comps.schlick();

        assert_abs_diff_eq!(reflectance, 1.0);
    }

    #[test]
    fn test_the_schlick_approximation_with_a_perpendicular_viewing_angle() {
        let shape = glass_sphere_builder(1.5).build();
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        let intersections = vec![
            Intersection::new(-1.0, &shape),
            Intersection::new(1.0, &shape),
        ];
        let comps =
            Computations::prepare_computations(&intersections[1], &ray, Some(&intersections))
                .unwrap();
        let reflectance = comps.schlick();

        assert_abs_diff_eq!(reflectance, 0.04);
    }

    #[test]
    fn test_the_schlick_approximation_with_small_angle_and_n2_greater_than_n1() {
        let shape = glass_sphere_builder(1.5).build();
        let ray = Ray::new(Point3::new(0.0, 0.99, -2.0), Vector3::new(0.0, 0.0, 1.0));
        let intersections = vec![Intersection::new(1.8589, &shape)];
        let comps =
            Computations::prepare_computations(&intersections[0], &ray, Some(&intersections))
                .unwrap();
        let reflectance = comps.schlick();

        assert_abs_diff_eq!(reflectance, 0.48873, epsilon = 1e-5);
    }

    #[test]
    fn test_shade_hit_with_a_reflective_transparent_material() {
        let mut world = World::default();
        let floor = Plane::builder()
            .material(Material {
                reflective: 0.5,
                transparency: 0.5,
                refractive_index: 1.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -1.0, 0.0)))
            .build();
        world.objects.push(Rc::new(floor));
        let ball = Sphere::builder()
            .material(Material {
                color: Color::new(1.0, 0.0, 0.0),
                ambient: 0.5,
                ..Default::default()
            })
            .transform(Transform::translation(Translation3::new(0.0, -3.5, -0.5)))
            .build();
        world.objects.push(Rc::new(ball));
        let ray = Ray::new(
            Point3::new(0.0, 0.0, -3.0),
            Vector3::new(
                0.0,
                -std::f64::consts::SQRT_2 / 2.0,
                std::f64::consts::SQRT_2 / 2.0,
            ),
        );
        let intersections = vec![Intersection::new(
            std::f64::consts::SQRT_2,
            world.objects[3].as_ref(),
        )];
        let comps =
            Computations::prepare_computations(&intersections[0], &ray, Some(&intersections))
                .unwrap();
        let color = world.shade_hit(&comps, MAX_RECURSION_DEPTH).unwrap();

        assert_eq!(color, Color::new(0.93391, 0.69643, 0.69243));
    }
}
