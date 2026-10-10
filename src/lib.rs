//! A ray tracer built in Rust following *The Ray Tracer Challenge* by Jamis Buck.
//!
//! Scenes combine [shapes](geometry::shapes), [materials](scene::material), and a
//! [point light](scene::light::PointLight). A [camera](rendering::camera::Camera)
//! traces rays into a [world](scene::world::World), producing a
//! [canvas](rendering::canvas::Canvas) that can be saved as a PPM image.
//!
//! # Quick start
//!
//! Render the default scene: two spheres and a reflective floor.
//!
//! ```
//! use merlin_rt::{rendering::camera::Camera, scene::world::World};
//! use nalgebra::{Point3, Vector3};
//!
//! let world = World::default();
//! let camera = Camera::new(
//!     40,
//!     30,
//!     std::f64::consts::FRAC_PI_3,
//!     World::view_transform(
//!         Point3::new(0.0, 1.5, -5.0),
//!         Point3::origin(),
//!         Vector3::y(),
//!     ),
//! );
//! let image = camera.render(&world)?;
//! assert_eq!(image.dimensions(), (40, 30));
//! // image.write_to_ppm(std::path::Path::new("scene.ppm"))?;
//! # Ok::<(), merlin_rt::rendering::camera::RenderError>(())
//! ```
//!
//! # Conventions
//!
//! - Geometry uses `nalgebra` points and vectors with `f64` components.
//! - Angles are in radians; an untransformed camera looks along negative Z.
//! - Shape transforms map object space to parent space. Transform
//!   [sequences](geometry::transforms::Transform::sequence) apply in listed order.
//! - Scenes use single-threaded shared ownership. Keep root
//!   [groups](geometry::shapes::Group) alive so children can resolve their parents.
//! - Rendering uses Phong lighting, hard shadows, and depth-limited reflection
//!   and refraction. It samples one ray per pixel; output has no gamma correction.

pub mod geometry;
pub mod rendering;
pub mod scene;

/// Shared tolerance for floating-point comparisons and surface offsets.
pub(crate) const EPSILON: f64 = 1e-5;
