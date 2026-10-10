use super::{Intersection, ParentLink, Shape, Triangle, TriangleBuilder};
use crate::{
    geometry::{
        ray::Ray,
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};
use nalgebra::{Point3, Vector3};

/// A triangle with object-space vertex normals interpolated using barycentric
/// weights. Normals are normalized after interpolation and transformation, not
/// individually on construction. Degenerate triangles produce no intersections.
#[derive(Debug, PartialEq)]
pub struct SmoothTriangle {
    triangle: Triangle,
    n1: Vector3<f64>,
    n2: Vector3<f64>,
    n3: Vector3<f64>,
}

impl SmoothTriangle {
    /// Creates a builder initialized with [`SmoothTriangle::default`].
    pub fn builder() -> SmoothTriangleBuilder {
        SmoothTriangleBuilder::default()
    }

    /// Creates a triangle from object-space vertices and corresponding normals.
    ///
    /// Normals are stored without normalization. Uses an identity transform and
    /// default material.
    pub fn new(
        p1: Point3<f64>,
        p2: Point3<f64>,
        p3: Point3<f64>,
        n1: Vector3<f64>,
        n2: Vector3<f64>,
        n3: Vector3<f64>,
    ) -> Self {
        Self {
            triangle: Triangle::new(p1, p2, p3),
            n1,
            n2,
            n3,
        }
    }

    /// Returns the first object-space vertex.
    pub fn p1(&self) -> Point3<f64> {
        self.triangle.p1()
    }

    /// Returns the second object-space vertex.
    pub fn p2(&self) -> Point3<f64> {
        self.triangle.p2()
    }

    /// Returns the third object-space vertex.
    pub fn p3(&self) -> Point3<f64> {
        self.triangle.p3()
    }

    /// Returns the object-space edge `p2 - p1`.
    pub fn e1(&self) -> Vector3<f64> {
        self.triangle.e1()
    }

    /// Returns the object-space edge `p3 - p1`.
    pub fn e2(&self) -> Vector3<f64> {
        self.triangle.e2()
    }

    /// Returns the stored object-space normal at the first vertex.
    pub fn n1(&self) -> Vector3<f64> {
        self.n1
    }

    /// Returns the stored object-space normal at the second vertex.
    pub fn n2(&self) -> Vector3<f64> {
        self.n2
    }

    /// Returns the stored object-space normal at the third vertex.
    pub fn n3(&self) -> Vector3<f64> {
        self.n3
    }

    fn interpolated_normal(&self, u: f64, v: f64) -> Result<Vector3<f64>, TransformError> {
        self.normal_to_world(self.n1 * (1.0 - u - v) + self.n2 * u + self.n3 * v)
    }
}

impl Default for SmoothTriangle {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// A builder for [`SmoothTriangle`].
///
/// Uses the default [`Triangle`] vertices and `(0, 0, -1)` for each normal, with
/// an identity transform and default material.
#[derive(Debug)]
#[must_use = "call build() to create the shape"]
pub struct SmoothTriangleBuilder {
    triangle: TriangleBuilder,
    n1: Vector3<f64>,
    n2: Vector3<f64>,
    n3: Vector3<f64>,
}

impl Default for SmoothTriangleBuilder {
    fn default() -> Self {
        Self {
            triangle: Triangle::builder(),
            n1: Vector3::new(0.0, 0.0, -1.0),
            n2: Vector3::new(0.0, 0.0, -1.0),
            n3: Vector3::new(0.0, 0.0, -1.0),
        }
    }
}

impl SmoothTriangleBuilder {
    /// Sets the object-space vertices and recomputes cached geometry.
    pub fn vertices(mut self, p1: Point3<f64>, p2: Point3<f64>, p3: Point3<f64>) -> Self {
        self.triangle = self.triangle.vertices(p1, p2, p3);
        self
    }

    /// Sets object-space normals for `p1`, `p2`, and `p3`, without normalizing them.
    pub fn normals(mut self, n1: Vector3<f64>, n2: Vector3<f64>, n3: Vector3<f64>) -> Self {
        self.n1 = n1;
        self.n2 = n2;
        self.n3 = n3;
        self
    }

    /// Sets the object-to-parent transform, replacing any previous transform.
    pub fn transform(mut self, transform: Transform) -> Self {
        self.triangle = self.triangle.transform(transform);
        self
    }

    /// Sets the surface material.
    pub fn material(mut self, material: Material) -> Self {
        self.triangle = self.triangle.material(material);
        self
    }

    /// Returns the configured shape.
    pub fn build(self) -> SmoothTriangle {
        SmoothTriangle {
            triangle: self.triangle.build(),
            n1: self.n1,
            n2: self.n2,
            n3: self.n3,
        }
    }
}

impl Shape for SmoothTriangle {
    fn parent_link(&self) -> &ParentLink {
        self.triangle.parent_link()
    }

    fn transform(&self) -> &Transform {
        self.triangle.transform()
    }

    fn material(&self) -> &Material {
        self.triangle.material()
    }

    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        Ok(self
            .triangle
            .intersect_uv(ray)?
            .map(|(t, u, v)| Intersection::with_uv(t, self, u, v))
            .into_iter()
            .collect())
    }

    /// Computes barycentric weights from the point when no hit is available.
    /// The rendering path uses `normal_at_hit` to reuse the weights from the ray.
    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        let offset = self.world_to_object(world_point)? - self.p1();
        let e1 = self.e1();
        let e2 = self.e2();
        let d11 = e1.dot(&e1);
        let d12 = e1.dot(&e2);
        let d22 = e2.dot(&e2);
        let denominator = d11 * d22 - d12 * d12;
        if denominator <= 0.0 {
            return self.normal_to_world(Vector3::zeros());
        }
        let d1 = offset.dot(&e1);
        let d2 = offset.dot(&e2);
        let u = (d22 * d1 - d12 * d2) / denominator;
        let v = (d11 * d2 - d12 * d1) / denominator;
        self.interpolated_normal(u, v)
    }

    fn normal_at_hit(
        &self,
        world_point: Point3<f64>,
        hit: &Intersection<'_>,
    ) -> Result<Vector3<f64>, TransformError> {
        match (hit.u, hit.v) {
            (Some(u), Some(v)) => self.interpolated_normal(u, v),
            _ => self.normal_at(world_point),
        }
    }
}
