use super::{Intersection, ParentLink, Shape, ShapeRef, SmoothTriangle, Triangle};
use crate::{
    geometry::{
        ray::Ray,
        transforms::{Transform, TransformError},
    },
    scene::material::Material,
};
use nalgebra::{Point3, Vector3};
use obj::raw::{object::Polygon, parse_obj};
use std::{fs::File, io::BufReader, path::Path, rc::Rc};
use thiserror::Error;

/// Invalid child ownership when constructing a group.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum GroupError {
    /// The same shared shape was supplied more than once.
    #[error("Child at index {index} occurs more than once in the group")]
    DuplicateChild {
        /// Zero-based index of the repeated child.
        index: usize,
    },
    /// A child already belongs to a group that is still alive.
    #[error("Child at index {index} already has a live parent")]
    ChildAlreadyHasParent {
        /// Zero-based index of the child with a live parent.
        index: usize,
    },
}

/// Errors encountered while loading an OBJ file into a group.
#[derive(Debug, Error)]
pub enum ObjImportError {
    /// The file could not be read or parsed as OBJ.
    #[error(transparent)]
    Obj(#[from] obj::ObjError),
    /// The imported group violated child ownership requirements.
    #[error(transparent)]
    Group(#[from] GroupError),
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
    /// Creates an empty group builder with identity transform and default material.
    pub fn builder() -> GroupBuilder {
        GroupBuilder::default()
    }

    /// Loads a Wavefront OBJ file as a group of flat or smooth triangles.
    ///
    /// Faces are fan-triangulated in file order, preserving vertex winding. As
    /// in the book, this assumes convex polygons; triangulate concave faces
    /// before importing. Faces with vertex normals become smooth triangles;
    /// faces without normals remain flat. OBJ groups are flattened, and texture
    /// coordinates, materials, points and lines are ignored. All shapes use
    /// identity transforms and default materials.
    ///
    /// # Errors
    ///
    /// Returns file I/O, OBJ parsing, or group construction errors.
    pub fn from_obj(path: impl AsRef<Path>) -> Result<Rc<Self>, ObjImportError> {
        let file = File::open(path).map_err(obj::ObjError::from)?;
        let raw = parse_obj(BufReader::new(file))?;
        let point = |index: usize| {
            let (x, y, z, _) = raw.positions[index];
            Point3::new(f64::from(x), f64::from(y), f64::from(z))
        };
        let normal = |index: usize| {
            let (x, y, z) = raw.normals[index];
            Vector3::new(f64::from(x), f64::from(y), f64::from(z))
        };
        let mut group = Self::builder();

        for polygon in raw.polygons {
            let indices: Vec<(usize, Option<usize>)> = match polygon {
                Polygon::P(indices) => indices.into_iter().map(|p| (p, None)).collect(),
                Polygon::PT(vertices) => vertices.into_iter().map(|(p, _)| (p, None)).collect(),
                Polygon::PN(vertices) => vertices.into_iter().map(|(p, n)| (p, Some(n))).collect(),
                Polygon::PTN(vertices) => {
                    vertices.into_iter().map(|(p, _, n)| (p, Some(n))).collect()
                }
            };

            // The parser guarantees at least three valid position indices.
            for edge in indices[1..].windows(2) {
                let [a, b, c] = [indices[0], edge[0], edge[1]];
                let child: ShapeRef = match (a.1, b.1, c.1) {
                    (Some(n1), Some(n2), Some(n3)) => Rc::new(SmoothTriangle::new(
                        point(a.0),
                        point(b.0),
                        point(c.0),
                        normal(n1),
                        normal(n2),
                        normal(n3),
                    )),
                    _ => Rc::new(Triangle::new(point(a.0), point(b.0), point(c.0))),
                };
                group = group.child(child);
            }
        }

        Ok(group.build()?)
    }

    /// Validates every child before attaching any parent links.
    ///
    /// A fresh group cannot already be among its descendants. Keeping children
    /// immutable and attachment private preserves an acyclic tree.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError`] for duplicate children or a child with a live parent.
    /// No parent links are changed on failure.
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

    /// Borrows the children in insertion order; the list cannot be mutated.
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

/// Builds an immutable group with shared children.
///
/// Defaults to no children, an identity transform, and a default material.
/// Keep the resulting group alive while using its descendants.
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
    /// Sets the transform from group space to parent space.
    pub fn transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Sets the group's own material; it is not inherited by its children.
    pub fn material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    /// Appends a child. Ownership is validated by [`Self::build`].
    pub fn child(mut self, child: ShapeRef) -> Self {
        self.children.push(child);
        self
    }

    /// Appends children in iterator order; does not replace existing children.
    pub fn children(mut self, children: impl IntoIterator<Item = ShapeRef>) -> Self {
        self.children.extend(children);
        self
    }

    /// Validates children, attaches weak parent links, and returns a shared group.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError`] for duplicate children or a child with a live parent.
    /// No parent links are changed on failure.
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

    fn load_obj_source(source: &str) -> Result<Rc<Group>, ObjImportError> {
        let path = std::env::temp_dir().join(format!("merlin-{}.obj", uuid::Uuid::new_v4()));
        std::fs::write(&path, source).unwrap();
        let result = Group::from_obj(&path);
        std::fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn test_loading_obj_triangles_and_flattening_groups() {
        let group = load_obj_source(
            "v 0 1 0\nv -1 0 0\nv 1 0 0\nv 0 1 2\nv -1 0 2\nv 1 0 2\n\
             g First\nf 1 2 3\ng Second\nf 4 5 6\n",
        )
        .unwrap();

        assert_eq!(group.children().len(), 2);
        assert_eq!(*group.transform(), Transform::identity());
        assert!(group.parent().is_none());
        let parent: ShapeRef = group.clone();
        for child in group.children() {
            assert!(Rc::ptr_eq(&child.parent().unwrap(), &parent));
            assert_eq!(*child.transform(), Transform::identity());
            assert_eq!(*child.material(), Material::default());
        }

        let ray = Ray::new(Point3::new(0.0, 0.5, -2.0), Vector3::new(0.0, 0.0, 1.0));
        let xs = group.intersect(&ray).unwrap();
        assert_eq!(xs.len(), 2);
        for (index, t) in [2.0, 4.0].into_iter().enumerate() {
            assert_abs_diff_eq!(xs[index].t, t);
            assert!(std::ptr::addr_eq(
                xs[index].object,
                group.children()[index].as_ref()
            ));
            assert_eq!(
                xs[index]
                    .object
                    .normal_at(ray.origin + ray.direction * t)
                    .unwrap(),
                Vector3::new(0.0, 0.0, -1.0)
            );
        }
        let miss = Ray::new(Point3::new(1.0, 1.0, -2.0), ray.direction);
        assert!(group.intersect(&miss).unwrap().is_empty());
    }

    #[test]
    fn test_loading_obj_face_formats_without_normals() {
        for face in ["1 2 3", "1/3 2/2 3/1"] {
            let group = load_obj_source(&format!(
                "v 0 1 0\nv -1 0 0\nv 1 0 0\n\
                 vt 0 0\nvt 1 0\nvt 0 1\nvn 1 0 0\nf {face}\n"
            ))
            .unwrap();
            assert_eq!(group.children().len(), 1);
            let ray = Ray::new(Point3::new(0.0, 0.5, -2.0), Vector3::new(0.0, 0.0, 1.0));
            let xs = group.intersect(&ray).unwrap();
            assert_eq!(xs.len(), 1);
            assert_abs_diff_eq!(xs[0].t, 2.0);
            assert_eq!(
                xs[0].object.normal_at(Point3::new(0.0, 0.5, 0.0)).unwrap(),
                Vector3::new(0.0, 0.0, -1.0)
            );
        }
    }

    #[test]
    fn test_fan_triangulating_an_obj_polygon_with_relative_indices() {
        let group =
            load_obj_source("v -1 1 0\nv -1 0 0\nv 1 0 0\nv 1 1 0\nv 0 2 0\nf -5 -4 -3 -2 -1\n")
                .unwrap();
        assert_eq!(group.children().len(), 3);

        // One ray through the interior of each triangle in the fan.
        for (index, (x, y)) in [(-0.5, 0.25), (0.5, 0.75), (0.0, 1.5)]
            .into_iter()
            .enumerate()
        {
            let ray = Ray::new(Point3::new(x, y, -2.0), Vector3::new(0.0, 0.0, 1.0));
            let xs = group.intersect(&ray).unwrap();
            assert_eq!(xs.len(), 1);
            assert_abs_diff_eq!(xs[0].t, 2.0);
            assert!(std::ptr::addr_eq(
                xs[0].object,
                group.children()[index].as_ref()
            ));
        }
    }

    #[test]
    fn test_loading_obj_without_faces() {
        for source in ["", "# no faces\nv 0 0 0\nv 1 0 0\np 1\nl 1 2\n"] {
            assert!(load_obj_source(source).unwrap().children().is_empty());
        }
    }

    #[test]
    fn test_loading_obj_reports_io_errors() {
        let path =
            std::env::temp_dir().join(format!("merlin-missing-{}.obj", uuid::Uuid::new_v4()));
        assert!(matches!(
            Group::from_obj(path),
            Err(ObjImportError::Obj(obj::ObjError::Io(error)))
                if error.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[test]
    fn test_loading_obj_reports_invalid_vertices_faces_and_indices() {
        for source in [
            "v invalid 0 0\n",
            "v 0 0 0\nf 1 1\n",
            "v 0 0 0\nf 0 1 1\n",
            "v 0 0 0\nf 1 2 3\n",
        ] {
            assert!(matches!(
                load_obj_source(source),
                Err(ObjImportError::Obj(_))
            ));
        }
    }

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
