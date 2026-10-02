use crate::{
    material::Material,
    ray::Ray,
    transforms::{Transform, TransformError},
};

use nalgebra::{Point3, Vector3};

pub trait Shape: Clone {
    fn transform(&self) -> Transform {
        Transform::identity()
    }

    fn material(&self) -> Material {
        Material::default()
    }

    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<Self>>, TransformError>;

    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError>;
}

#[derive(Debug)]
pub struct Intersection<T: Shape> {
    pub t: f64,
    pub object: T,
}

impl<T: Shape> Intersection<T> {
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
