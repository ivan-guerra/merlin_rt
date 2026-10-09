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

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GroupError {
    #[error("Child at index {index} occurs more than once in the group")]
    DuplicateChild { index: usize },
    #[error("Child at index {index} already has a live parent")]
    ChildAlreadyHasParent { index: usize },
}

/// A shape that owns an immutable list of children and has no surface of its own.
///
/// Build groups bottom-up. Each child may have only one live parent. Children
/// own only a weak parent link, so retaining a child does not keep its group alive.
/// Group materials satisfy the `Shape` interface; hits use each leaf's material.
#[derive(Debug)]
pub struct Group {
    transform: Transform,
    material: Material,
    parent: ParentLink,
    children: Vec<ShapeRef>,
}

impl Group {
    pub fn builder() -> GroupBuilder {
        GroupBuilder::default()
    }

    /// Validates every child before attaching any parent links.
    ///
    /// A fresh group cannot already be among its descendants. Keeping children
    /// immutable and attachment private preserves an acyclic tree.
    pub fn new(
        transform: Transform,
        material: Material,
        children: Vec<ShapeRef>,
    ) -> Result<Rc<Self>, GroupError> {
        for (index, child) in children.iter().enumerate() {
            if children[..index]
                .iter()
                .any(|other| Rc::ptr_eq(child, other))
            {
                return Err(GroupError::DuplicateChild { index });
            }
            if child.parent().is_some() {
                return Err(GroupError::ChildAlreadyHasParent { index });
            }
        }

        let group = Rc::new(Self {
            transform,
            material,
            parent: ParentLink::default(),
            children,
        });
        let parent: ShapeRef = group.clone();
        for child in &group.children {
            child.parent_link().attach(&parent);
        }
        Ok(group)
    }

    pub fn children(&self) -> &[ShapeRef] {
        &self.children
    }
}

impl Shape for Group {
    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn parent_link(&self) -> &ParentLink {
        &self.parent
    }

    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        let ray = Ray::new(
            self.transform.apply_inverse(ray.origin)?,
            self.transform.apply_inverse(ray.direction)?,
        );
        let mut intersections = Vec::new();
        for child in &self.children {
            intersections.extend(child.intersect(&ray)?);
        }
        intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
        Ok(intersections)
    }

    fn normal_at(&self, _world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        Err(TransformError::UndefinedNormal)
    }
}

#[derive(Debug)]
#[must_use = "call build() to create the group"]
pub struct GroupBuilder {
    transform: Transform,
    material: Material,
    children: Vec<ShapeRef>,
}

impl Default for GroupBuilder {
    fn default() -> Self {
        Self {
            transform: Transform::identity(),
            material: Material::default(),
            children: Vec::new(),
        }
    }
}

impl GroupBuilder {
    pub fn transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    pub fn material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    pub fn child(mut self, child: ShapeRef) -> Self {
        self.children.push(child);
        self
    }

    pub fn children(mut self, children: impl IntoIterator<Item = ShapeRef>) -> Self {
        self.children.extend(children);
        self
    }

    pub fn build(self) -> Result<Rc<Group>, GroupError> {
        Group::new(self.transform, self.material, self.children)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{shapes::Sphere, transforms::Axis};
    use approx::assert_abs_diff_eq;
    use nalgebra::{Scale3, Translation3};
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_creating_a_new_group() {
        let g = Group::builder().build().unwrap();

        assert_eq!(*g.transform(), Transform::identity());
        assert!(g.children().is_empty());
    }

    #[test]
    fn test_a_shape_has_a_parent_attribute() {
        // A sphere is sufficient to exercise the shared Shape interface.
        let s = Sphere::default();

        assert!(s.parent().is_none());
    }

    #[test]
    fn test_adding_a_child_to_a_group() {
        let s: ShapeRef = Rc::new(Sphere::default());
        // Children are attached during construction, not by mutating a live group.
        let g = Group::builder().child(s.clone()).build().unwrap();

        assert!(!g.children().is_empty());
        assert!(g.children().iter().any(|child| Rc::ptr_eq(child, &s)));
        let parent: ShapeRef = g.clone();
        assert!(Rc::ptr_eq(&s.parent().unwrap(), &parent));
    }

    #[test]
    fn test_intersecting_a_ray_with_an_empty_group() {
        let g = Group::builder().build().unwrap();
        let r = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));

        // With an identity transform, intersect is equivalent to local_intersect.
        let xs = g.intersect(&r).unwrap();

        assert!(xs.is_empty());
    }

    #[test]
    fn test_intersecting_a_ray_with_a_nonempty_group() {
        let s1: ShapeRef = Rc::new(Sphere::default());
        let s2: ShapeRef = Rc::new(
            Sphere::builder()
                .transform(Transform::translation(Translation3::new(0.0, 0.0, -3.0)))
                .build(),
        );
        let s3: ShapeRef = Rc::new(
            Sphere::builder()
                .transform(Transform::translation(Translation3::new(5.0, 0.0, 0.0)))
                .build(),
        );
        let g = Group::builder()
            .children([s1.clone(), s2.clone(), s3])
            .build()
            .unwrap();
        let r = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));

        let xs = g.intersect(&r).unwrap();

        assert_eq!(xs.len(), 4);
        for (intersection, (t, object)) in xs.iter().zip([
            (1.0, s2.as_ref()),
            (3.0, s2.as_ref()),
            (4.0, s1.as_ref()),
            (6.0, s1.as_ref()),
        ]) {
            assert_abs_diff_eq!(intersection.t, t);
            assert!(std::ptr::addr_eq(intersection.object, object));
        }
    }

    #[test]
    fn test_intersecting_a_transformed_group() {
        let s: ShapeRef = Rc::new(
            Sphere::builder()
                .transform(Transform::translation(Translation3::new(5.0, 0.0, 0.0)))
                .build(),
        );
        let g = Group::builder()
            .transform(Transform::scale(Scale3::new(2.0, 2.0, 2.0)))
            .child(s.clone())
            .build()
            .unwrap();
        let r = Ray::new(Point3::new(10.0, 0.0, -10.0), Vector3::new(0.0, 0.0, 1.0));

        let xs = g.intersect(&r).unwrap();

        assert_eq!(xs.len(), 2);
        assert_abs_diff_eq!(xs[0].t, 8.0);
        assert_abs_diff_eq!(xs[1].t, 12.0);
        for intersection in xs {
            assert!(std::ptr::addr_eq(intersection.object, s.as_ref()));
        }
    }

    #[test]
    fn test_converting_a_point_from_world_to_object_space() {
        let s = Rc::new(
            Sphere::builder()
                .transform(Transform::translation(Translation3::new(5.0, 0.0, 0.0)))
                .build(),
        );
        let g2 = Group::builder()
            .transform(Transform::scale(Scale3::new(2.0, 2.0, 2.0)))
            .child(s.clone())
            .build()
            .unwrap();
        // Keep the root alive while following the weak parent links.
        let _g1 = Group::builder()
            .transform(Transform::rotation(Axis::Y, FRAC_PI_2))
            .child(g2)
            .build()
            .unwrap();

        let p = s.world_to_object(Point3::new(-2.0, 0.0, -10.0)).unwrap();

        assert_abs_diff_eq!(p, Point3::new(0.0, 0.0, -1.0), epsilon = 1e-10);
    }

    #[test]
    fn test_converting_a_normal_from_object_to_world_space() {
        let s = Rc::new(
            Sphere::builder()
                .transform(Transform::translation(Translation3::new(5.0, 0.0, 0.0)))
                .build(),
        );
        let g2 = Group::builder()
            .transform(Transform::scale(Scale3::new(1.0, 2.0, 3.0)))
            .child(s.clone())
            .build()
            .unwrap();
        let _g1 = Group::builder()
            .transform(Transform::rotation(Axis::Y, FRAC_PI_2))
            .child(g2)
            .build()
            .unwrap();
        let component = 3.0_f64.sqrt() / 3.0;

        let n = s
            .normal_to_world(Vector3::new(component, component, component))
            .unwrap();

        // The book's expected normal is rounded to four decimal places.
        assert_abs_diff_eq!(n, Vector3::new(0.2857, 0.4286, -0.8571), epsilon = 1e-4);
        assert_abs_diff_eq!(n.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_finding_the_normal_on_a_child_object() {
        let s = Rc::new(
            Sphere::builder()
                .transform(Transform::translation(Translation3::new(5.0, 0.0, 0.0)))
                .build(),
        );
        let g2 = Group::builder()
            .transform(Transform::scale(Scale3::new(1.0, 2.0, 3.0)))
            .child(s.clone())
            .build()
            .unwrap();
        let _g1 = Group::builder()
            .transform(Transform::rotation(Axis::Y, FRAC_PI_2))
            .child(g2)
            .build()
            .unwrap();

        let n = s.normal_at(Point3::new(1.7321, 1.1547, -5.5774)).unwrap();

        assert_abs_diff_eq!(n, Vector3::new(0.2857, 0.4286, -0.8571), epsilon = 1e-4);
        assert_abs_diff_eq!(n.norm(), 1.0, epsilon = 1e-10);
    }
}
