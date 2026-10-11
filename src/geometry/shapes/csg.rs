use super::{Intersection, ParentLink, Shape, ShapeRef};
use crate::{
    geometry::{
        ray::Ray,
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};
use nalgebra::{Point3, Vector3};
use std::rc::Rc;
use thiserror::Error;

/// The boolean operation applied to a CSG node's left and right operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsgOperation {
    /// Keep the volume inside either operand.
    Union,
    /// Keep only the volume inside both operands.
    Intersection,
    /// Subtract the right operand's volume from the left operand.
    Difference,
}

impl CsgOperation {
    /// Reports whether a boundary crossing belongs to the combined surface.
    ///
    /// `left_hit` identifies the operand containing the hit leaf. `inside_left`
    /// and `inside_right` describe the ray immediately before this crossing.
    pub fn intersection_allowed(
        self,
        left_hit: bool,
        inside_left: bool,
        inside_right: bool,
    ) -> bool {
        match self {
            Self::Union => (left_hit && !inside_right) || (!left_hit && !inside_left),
            Self::Intersection => (left_hit && inside_right) || (!left_hit && inside_left),
            Self::Difference => (left_hit && !inside_right) || (!left_hit && inside_left),
        }
    }
}

/// Invalid child ownership when constructing a CSG node.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CsgError {
    /// Both operands refer to the same shared shape.
    #[error("CSG operands must be distinct shapes")]
    DuplicateChild,
    /// An operand already belongs to a live group or CSG node.
    #[error("CSG child at index {index} already has a live parent")]
    ChildAlreadyHasParent {
        /// Zero for the left operand, one for the right operand.
        index: usize,
    },
}

/// An immutable constructive solid geometry node combining two shared shapes.
///
/// Operands may be primitives, groups, or other CSG nodes. Build trees bottom-up;
/// each child can have only one live parent. Keep the root alive while using its
/// descendants, whose parent links are weak.
///
/// Filtering follows the book's inside/outside crossing rules. Operands should
/// describe closed solids with alternating entry and exit intersections; open
/// surfaces and overlapping group children do not necessarily satisfy this.
/// Hits retain their leaf shapes, materials, normals, and barycentric weights.
/// The CSG node has no normal of its own and its material is not inherited.
///
/// ```
/// use merlin_rt::geometry::shapes::{Csg, CsgOperation, Cube, Sphere};
/// use std::rc::Rc;
///
/// let shape = Csg::new(
///     CsgOperation::Difference,
///     Rc::new(Cube::default()),
///     Rc::new(Sphere::builder().radius(1.3).build()),
/// )?;
/// // Insert `shape` into a Group or World::from_shared.
/// # Ok::<(), merlin_rt::geometry::shapes::CsgError>(())
/// ```
#[derive(Debug)]
pub struct Csg {
    operation: CsgOperation,
    left: ShapeRef,
    right: ShapeRef,
    transform: Transform,
    material: Material,
    parent: ParentLink,
}

impl Csg {
    /// Creates a builder with the required operands, identity transform, and default material.
    pub fn builder(operation: CsgOperation, left: ShapeRef, right: ShapeRef) -> CsgBuilder {
        CsgBuilder {
            operation,
            left,
            right,
            transform: Transform::identity(),
            material: Material::default(),
        }
    }

    /// Creates a CSG node with identity transform and default material.
    ///
    /// # Errors
    ///
    /// Returns [`CsgError`] for duplicate operands or an operand with a live parent.
    /// No parent links are changed on failure.
    pub fn new(
        operation: CsgOperation,
        left: ShapeRef,
        right: ShapeRef,
    ) -> Result<Rc<Self>, CsgError> {
        Self::builder(operation, left, right).build()
    }

    /// Returns the boolean operation.
    pub fn operation(&self) -> CsgOperation {
        self.operation
    }

    /// Borrows the left operand.
    pub fn left(&self) -> &ShapeRef {
        &self.left
    }

    /// Borrows the right operand.
    pub fn right(&self) -> &ShapeRef {
        &self.right
    }

    /// Filters the combined operand intersections, preserving the original hits.
    ///
    /// Supply all hits from these operands in ascending `t` order, including
    /// negative hits so rays originating inside an operand are handled correctly.
    /// The ray starts outside both operands at negative infinity.
    pub fn filter_intersections<'a>(
        &self,
        intersections: Vec<Intersection<'a>>,
    ) -> Vec<Intersection<'a>> {
        let mut inside_left = false;
        let mut inside_right = false;
        intersections
            .into_iter()
            .filter(|hit| {
                let left_hit = self.left.includes(hit.object);
                let allowed =
                    self.operation
                        .intersection_allowed(left_hit, inside_left, inside_right);
                // Even a discarded boundary changes the operand's inside state.
                if left_hit {
                    inside_left = !inside_left;
                } else {
                    inside_right = !inside_right;
                }
                allowed
            })
            .collect()
    }
}

impl Shape for Csg {
    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn parent_link(&self) -> &ParentLink {
        &self.parent
    }

    fn includes(&self, other: &dyn Shape) -> bool {
        std::ptr::addr_eq(self, other) || self.left.includes(other) || self.right.includes(other)
    }

    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        let ray = Ray::new(
            self.transform.apply_inverse(ray.origin)?,
            self.transform.apply_inverse(ray.direction)?,
        );
        let mut intersections = self.left.intersect(&ray)?;
        intersections.extend(self.right.intersect(&ray)?);
        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
        Ok(self.filter_intersections(intersections))
    }

    fn normal_at(&self, _world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        Err(TransformError::UndefinedNormal)
    }
}

/// Builds a CSG node with immutable operands and weak parent links.
///
/// Created by [`Csg::builder`] with an operation and both operands; defaults to
/// an identity transform and default material. Ownership is checked on build.
#[derive(Debug)]
#[must_use = "call build() to create the CSG node"]
pub struct CsgBuilder {
    operation: CsgOperation,
    left: ShapeRef,
    right: ShapeRef,
    transform: Transform,
    material: Material,
}

impl CsgBuilder {
    /// Sets the transform from CSG space to parent space.
    pub fn transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Sets the node's own material; it is not inherited by its children.
    pub fn material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    /// Validates both operands, attaches weak parent links, and returns a shared node.
    ///
    /// # Errors
    ///
    /// Returns [`CsgError`] for duplicate operands or an operand with a live parent.
    /// No parent links are changed on failure.
    pub fn build(self) -> Result<Rc<Csg>, CsgError> {
        if Rc::ptr_eq(&self.left, &self.right) {
            return Err(CsgError::DuplicateChild);
        }
        for (index, child) in [&self.left, &self.right].into_iter().enumerate() {
            if child.parent().is_some() {
                return Err(CsgError::ChildAlreadyHasParent { index });
            }
        }

        let csg = Rc::new(Csg {
            operation: self.operation,
            left: self.left,
            right: self.right,
            transform: self.transform,
            material: self.material,
            parent: ParentLink::default(),
        });
        let parent: ShapeRef = csg.clone();
        csg.left.parent_link().attach(&parent);
        csg.right.parent_link().attach(&parent);
        Ok(csg)
    }
}
