mod cube;
mod cylinder;
mod double_napped_cone;
mod plane;
mod sphere;

pub use cube::Cube;
pub use cylinder::Cylinder;
pub use double_napped_cone::DoubleNappedCone;
pub use plane::Plane;
pub use sphere::Sphere;

use crate::{
    light::Lighting,
    material::Material,
    ray::Ray,
    transforms::{Transform, TransformError},
};
use nalgebra::{Point3, Vector3};
use std::fmt::Debug;

pub trait Shape: Debug + Lighting {
    fn transform(&self) -> &Transform;
    fn material(&self) -> &Material;
    fn material_mut(&mut self) -> &mut Material;
    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError>;
    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError>;
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
