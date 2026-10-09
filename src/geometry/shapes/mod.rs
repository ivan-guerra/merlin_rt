mod cube;
mod cylinder;
mod double_napped_cone;
mod plane;
mod sphere;

pub use cube::{Cube, CubeBuilder};
pub use cylinder::{Cylinder, CylinderBuilder};
pub use double_napped_cone::{DoubleNappedCone, DoubleNappedConeBuilder};
pub use plane::{Plane, PlaneBuilder};
pub use sphere::{Sphere, SphereBuilder};

use crate::{
    geometry::{
        ray::Ray,
        transforms::{Transform, TransformError},
    },
    rendering::canvas::Color,
    scene::{light::PointLight, material::Material},
};
use nalgebra::{Point3, Vector3};
use std::fmt::Debug;

pub trait Shape: Debug {
    fn transform(&self) -> &Transform;
    fn material(&self) -> &Material;
    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError>;
    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError>;
    fn lighting(
        &self,
        light: PointLight,
        point: Point3<f64>,
        eyev: Vector3<f64>,
        normalv: Vector3<f64>,
        in_shadow: bool,
    ) -> Result<Color, TransformError> {
        let material = self.material();
        let color = match &material.pattern {
            Some(pattern) => pattern.pattern_at_object(self.transform(), point)?,
            None => material.color,
        };

        let effective_color = color * light.intensity;
        let ambient = effective_color * material.ambient;

        if in_shadow {
            return Ok(ambient);
        }

        let lightv = (light.position - point).normalize();
        let light_dot_normal = lightv.dot(&normalv);

        if light_dot_normal < 0.0 {
            return Ok(ambient);
        }

        let diffuse = effective_color * material.diffuse * light_dot_normal;
        let reflectv = Transform::reflection(normalv).apply(-lightv);
        let reflect_dot_eye = reflectv.dot(&eyev);

        let specular = if reflect_dot_eye <= 0.0 {
            Color::new(0.0, 0.0, 0.0)
        } else {
            light.intensity * material.specular * reflect_dot_eye.powf(material.shininess)
        };

        Ok(ambient + diffuse + specular)
    }
}

#[derive(Debug)]
pub struct Intersection<'a> {
    pub t: f64,
    pub object: &'a dyn Shape,
}

impl<'a> Intersection<'a> {
    pub fn new(t: f64, object: &'a dyn Shape) -> Self {
        Self { t, object }
    }

    pub fn hit(intersections: &mut [Self]) -> Option<&Self> {
        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
        intersections
            .iter()
            .find(|intersection| intersection.t >= 0.0)
    }
}

impl PartialEq for Intersection<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t && std::ptr::addr_eq(self.object, other.object)
    }
}
