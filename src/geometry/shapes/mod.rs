//! Surface primitives, hierarchical groups, and ray intersections.
//!
//! Construct shapes with their builders, then place them in a
//! [`World`](crate::scene::world::World). Builders start with identity transforms
//! and default materials. Groups own shared children; their transforms compose
//! through parent links, but their materials are not inherited.
//!
//! [`Shape::intersect`] accepts a ray in parent space, while normals and lighting
//! use world-space inputs. Singular transforms return [`TransformError`].

mod cube;
mod cylinder;
mod double_napped_cone;
mod group;
mod plane;
mod smooth_triangle;
mod sphere;
mod triangle;

pub use cube::{Cube, CubeBuilder};
pub use cylinder::{Cylinder, CylinderBuilder};
pub use double_napped_cone::{DoubleNappedCone, DoubleNappedConeBuilder};
pub use group::{Group, GroupBuilder, GroupError, ObjImportError};
pub use plane::{Plane, PlaneBuilder};
pub use smooth_triangle::{SmoothTriangle, SmoothTriangleBuilder};
pub use sphere::{Sphere, SphereBuilder};
pub use triangle::{Triangle, TriangleBuilder};

use crate::{
    geometry::{
        ray::Ray,
        transforms::{Transform, TransformError},
    },
    rendering::canvas::Color,
    scene::{light::PointLight, material::Material},
};
use nalgebra::{Point3, Vector3};
use std::{
    cell::RefCell,
    fmt::Debug,
    rc::{Rc, Weak},
};

/// Shared ownership of a shape in a single-threaded scene graph.
pub type ShapeRef = Rc<dyn Shape>;

/// A non-owning parent link. Only group construction can attach a parent.
#[derive(Debug, Default)]
pub struct ParentLink {
    parent: RefCell<Option<Weak<dyn Shape>>>,
}

impl ParentLink {
    /// Returns `None` for a root shape or when its parent has been dropped.
    pub fn resolve(&self) -> Option<ShapeRef> {
        self.parent.borrow().as_ref().and_then(Weak::upgrade)
    }

    fn attach(&self, parent: &ShapeRef) {
        *self.parent.borrow_mut() = Some(Rc::downgrade(parent));
    }
}

impl PartialEq for ParentLink {
    fn eq(&self, other: &Self) -> bool {
        match (&*self.parent.borrow(), &*other.parent.borrow()) {
            (None, None) => true,
            (Some(a), Some(b)) => Weak::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// A renderable surface or a group of child shapes.
///
/// Implementations expose a stable parent link and convert parent-space rays
/// to object space. Use [`Self::world_to_object`] and [`Self::normal_to_world`]
/// when computing normals so ancestor transforms are respected.
pub trait Shape: Debug {
    /// Returns the transform from object space to parent space.
    fn transform(&self) -> &Transform;
    /// Returns this shape's material; group materials do not affect children.
    fn material(&self) -> &Material;
    /// Must return this shape's own, stable parent link.
    fn parent_link(&self) -> &ParentLink;

    /// Resolves the parent, or returns `None` for a root or dropped parent.
    fn parent(&self) -> Option<ShapeRef> {
        self.parent_link().resolve()
    }

    /// Converts a world-space point through the entire ancestor chain.
    ///
    /// # Errors
    ///
    /// Returns an error if this shape or an ancestor has a singular transform.
    fn world_to_object(&self, point: Point3<f64>) -> Result<Point3<f64>, TransformError> {
        let point = match self.parent() {
            Some(parent) => parent.world_to_object(point)?,
            None => point,
        };
        self.transform().apply_inverse(point)
    }

    /// Converts an object-space normal through the entire ancestor chain.
    ///
    /// Nonzero normals are normalized after each inverse-transpose transform.
    ///
    /// # Errors
    ///
    /// Returns an error if this shape or an ancestor has a singular transform.
    fn normal_to_world(&self, normal: Vector3<f64>) -> Result<Vector3<f64>, TransformError> {
        let normal = self.transform().apply_transpose_inverse(normal)?;
        // Preserve a zero normal (for example, at a cone's apex).
        let normal = normal.try_normalize(0.0).unwrap_or(normal);
        match self.parent() {
            Some(parent) => parent.normal_to_world(normal),
            None => Ok(normal),
        }
    }

    /// Intersects a parent-space ray with this shape.
    ///
    /// Parent space is world space for a root shape. Apply only this shape's
    /// inverse transform, not its ancestors' transforms.
    ///
    /// # Errors
    ///
    /// Returns an error if a transform needed for intersection is singular.
    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError>;
    /// Returns a world-space surface normal at `world_point`.
    ///
    /// # Errors
    ///
    /// Returns an error for a singular transform or a group with no surface.
    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError>;

    /// Computes a world-space normal using any per-intersection data.
    /// Shapes without interpolated normals can ignore the hit.
    ///
    /// # Errors
    ///
    /// Returns an error for a singular transform or a group with no surface.
    fn normal_at_hit(
        &self,
        world_point: Point3<f64>,
        _hit: &Intersection<'_>,
    ) -> Result<Vector3<f64>, TransformError> {
        self.normal_at(world_point)
    }

    /// Evaluates Phong lighting with world-space inputs and unit eye/normal vectors.
    ///
    /// Patterns override the material color. A shadowed point receives ambient
    /// light only; reflection and refraction are handled by the world.
    ///
    /// # Errors
    ///
    /// Returns an error if a transform needed to sample a pattern is singular.
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
            Some(pattern) => {
                let object_point = self.world_to_object(point)?;
                // Already in object space; only the pattern transform remains.
                pattern.pattern_at_object(&Transform::identity(), object_point)?
            }
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

/// A ray parameter and the intersected leaf shape, with optional triangle weights.
#[derive(Debug)]
pub struct Intersection<'a> {
    /// Ray parameter; negative values lie behind the ray origin.
    pub t: f64,
    /// The intersected shape, borrowed from the scene.
    pub object: &'a dyn Shape,
    /// Barycentric weight of a triangle's second vertex (not a texture coordinate).
    pub u: Option<f64>,
    /// Barycentric weight of a triangle's third vertex (not a texture coordinate).
    pub v: Option<f64>,
}

impl<'a> Intersection<'a> {
    /// Creates an intersection without barycentric weights.
    pub fn new(t: f64, object: &'a dyn Shape) -> Self {
        Self {
            t,
            object,
            u: None,
            v: None,
        }
    }

    /// Records the barycentric weights used to interpolate triangle normals.
    pub fn with_uv(t: f64, object: &'a dyn Shape, u: f64, v: f64) -> Self {
        Self {
            t,
            object,
            u: Some(u),
            v: Some(v),
        }
    }

    /// Sorts intersections in place by `t` and returns the first with `t >= 0`.
    ///
    /// Returns `None` when no intersection is in front of the ray origin.
    pub fn hit(intersections: &mut [Self]) -> Option<&Self> {
        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
        intersections
            .iter()
            .find(|intersection| intersection.t >= 0.0)
    }
}

impl PartialEq for Intersection<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t
            && std::ptr::addr_eq(self.object, other.object)
            && self.u == other.u
            && self.v == other.v
    }
}
