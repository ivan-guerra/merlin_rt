pub mod geometry;
pub mod rendering;
pub mod scene;

/// Shared tolerance for floating-point comparisons and surface offsets.
pub(crate) const EPSILON: f64 = 1e-5;
