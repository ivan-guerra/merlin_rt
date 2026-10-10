//! Two-color procedural patterns with independent spatial transforms.
//!
//! All built-in patterns default to white (`a`), black (`b`), and an identity
//! transform. Sampling occurs in pattern space, after undoing the object and
//! pattern transforms.

use crate::{
    EPSILON,
    geometry::transforms::{Transform, TransformError},
    rendering::canvas::Color,
};

use nalgebra::Point3;
use std::fmt::Debug;

const BLACK: Color = Color::new(0.0, 0.0, 0.0);
const WHITE: Color = Color::new(1.0, 1.0, 1.0);

fn stable_floor(value: f64) -> i32 {
    if value.abs() < EPSILON {
        0
    } else {
        value.floor() as i32
    }
}

/// A procedural color source sampled in pattern-local coordinates.
pub trait Pattern: Debug {
    /// Samples a pattern-local point without applying transforms.
    ///
    /// Despite the parameter name, `world_point` must already be in pattern space.
    fn pattern_at(&self, world_point: Point3<f64>) -> Color;
    /// Samples a world-space point after undoing the object and pattern transforms.
    ///
    /// For grouped shapes, convert through the ancestor chain first and pass
    /// an identity object transform, as [`Shape::lighting`](crate::geometry::shapes::Shape::lighting) does.
    ///
    /// # Errors
    ///
    /// Returns an error if the object or pattern transform is singular.
    fn pattern_at_object(
        &self,
        object_transform: &Transform,
        world_point: Point3<f64>,
    ) -> Result<Color, TransformError>;
}

/// Alternating unit-width stripes along the pattern-space X axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StripePattern {
    /// First color; defaults to white.
    pub a: Color,
    /// Second color; defaults to black.
    pub b: Color,
    /// Pattern-to-object transform; defaults to identity.
    pub transform: Transform,
}

impl Default for StripePattern {
    fn default() -> Self {
        StripePattern {
            a: WHITE,
            b: BLACK,
            transform: Transform::identity(),
        }
    }
}

impl Pattern for StripePattern {
    fn pattern_at(&self, world_point: Point3<f64>) -> Color {
        if world_point.x.floor() as i32 % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }

    fn pattern_at_object(
        &self,
        object_transform: &Transform,
        world_point: Point3<f64>,
    ) -> Result<Color, TransformError> {
        let object_point = object_transform.apply_inverse(world_point)?;
        let pattern_point = self.transform.apply_inverse(object_point)?;
        Ok(self.pattern_at(pattern_point))
    }
}

/// A linear gradient from `a` toward `b`, repeating each unit along X.
#[derive(Debug)]
pub struct GradientPattern {
    /// First color; defaults to white.
    pub a: Color,
    /// Second color; defaults to black.
    pub b: Color,
    /// Pattern-to-object transform; defaults to identity.
    pub transform: Transform,
}

impl Default for GradientPattern {
    fn default() -> Self {
        GradientPattern {
            a: WHITE,
            b: BLACK,
            transform: Transform::identity(),
        }
    }
}

impl Pattern for GradientPattern {
    fn pattern_at(&self, world_point: Point3<f64>) -> Color {
        let distance = self.b - self.a;
        let fraction = world_point.x - world_point.x.floor();
        self.a + distance * fraction
    }

    fn pattern_at_object(
        &self,
        object_transform: &Transform,
        world_point: Point3<f64>,
    ) -> Result<Color, TransformError> {
        let object_point = object_transform.apply_inverse(world_point)?;
        let pattern_point = self.transform.apply_inverse(object_point)?;
        Ok(self.pattern_at(pattern_point))
    }
}

/// Alternating unit-width rings around the Y axis in the pattern-space XZ plane.
#[derive(Debug)]
pub struct RingPattern {
    /// First color; defaults to white.
    pub a: Color,
    /// Second color; defaults to black.
    pub b: Color,
    /// Pattern-to-object transform; defaults to identity.
    pub transform: Transform,
}

impl Default for RingPattern {
    fn default() -> Self {
        RingPattern {
            a: WHITE,
            b: BLACK,
            transform: Transform::identity(),
        }
    }
}

impl Pattern for RingPattern {
    fn pattern_at(&self, world_point: Point3<f64>) -> Color {
        let distance = (world_point.x.powi(2) + world_point.z.powi(2)).sqrt();
        if distance.floor() as i32 % 2 == 0 {
            self.a
        } else {
            self.b
        }
    }

    fn pattern_at_object(
        &self,
        object_transform: &Transform,
        world_point: Point3<f64>,
    ) -> Result<Color, TransformError> {
        let object_point = object_transform.apply_inverse(world_point)?;
        let pattern_point = self.transform.apply_inverse(object_point)?;
        Ok(self.pattern_at(pattern_point))
    }
}

/// A three-dimensional checker pattern alternating colors in unit cubes.
#[derive(Debug)]
pub struct CheckerPattern {
    /// First color; defaults to white.
    pub a: Color,
    /// Second color; defaults to black.
    pub b: Color,
    /// Pattern-to-object transform; defaults to identity.
    pub transform: Transform,
}

impl Default for CheckerPattern {
    fn default() -> Self {
        CheckerPattern {
            a: WHITE,
            b: BLACK,
            transform: Transform::identity(),
        }
    }
}

impl Pattern for CheckerPattern {
    fn pattern_at(&self, world_point: Point3<f64>) -> Color {
        if (stable_floor(world_point.x) + stable_floor(world_point.y) + stable_floor(world_point.z))
            % 2
            == 0
        {
            self.a
        } else {
            self.b
        }
    }

    fn pattern_at_object(
        &self,
        object_transform: &Transform,
        world_point: Point3<f64>,
    ) -> Result<Color, TransformError> {
        let object_point = object_transform.apply_inverse(world_point)?;
        let pattern_point = self.transform.apply_inverse(object_point)?;
        Ok(self.pattern_at(pattern_point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_a_stripe_pattern() {
        let pattern = StripePattern::default();
        assert_eq!(pattern.a, WHITE);
        assert_eq!(pattern.b, BLACK);
    }

    #[test]
    fn test_a_stripe_pattern_is_constant_in_y() {
        let pattern = StripePattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 1.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 2.0, 0.0)), WHITE);
    }

    #[test]
    fn test_a_stripe_pattern_is_constant_in_z() {
        let pattern = StripePattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 1.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 2.0)), WHITE);
    }

    #[test]
    fn test_a_stripe_pattern_alternates_in_x() {
        let pattern = StripePattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.9, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(1.0, 0.0, 0.0)), BLACK);
        assert_eq!(pattern.pattern_at(Point3::new(-0.1, 0.0, 0.0)), BLACK);
        assert_eq!(pattern.pattern_at(Point3::new(-1.0, 0.0, 0.0)), BLACK);
        assert_eq!(pattern.pattern_at(Point3::new(-1.1, 0.0, 0.0)), WHITE);
    }

    #[test]
    fn test_a_gradient_linearly_interpolates_between_colors() {
        let pattern = GradientPattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(
            pattern.pattern_at(Point3::new(0.25, 0.0, 0.0)),
            Color::new(0.75, 0.75, 0.75)
        );
        assert_eq!(
            pattern.pattern_at(Point3::new(0.5, 0.0, 0.0)),
            Color::new(0.5, 0.5, 0.5)
        );
        assert_eq!(
            pattern.pattern_at(Point3::new(0.75, 0.0, 0.0)),
            Color::new(0.25, 0.25, 0.25)
        );
    }

    #[test]
    fn test_a_ring_should_extend_in_both_x_and_z() {
        let pattern = RingPattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(1.0, 0.0, 0.0)), BLACK);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 1.0)), BLACK);
        assert_eq!(pattern.pattern_at(Point3::new(0.708, 0.0, 0.708)), BLACK);
    }

    #[test]
    fn test_checkers_should_repeat_in_x() {
        let pattern = CheckerPattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.99, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(1.01, 0.0, 0.0)), BLACK);
    }

    #[test]
    fn test_checkers_should_repeat_in_y() {
        let pattern = CheckerPattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.99, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 1.01, 0.0)), BLACK);
    }

    #[test]
    fn test_checkers_should_repeat_in_z() {
        let pattern = CheckerPattern::default();
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.0)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 0.99)), WHITE);
        assert_eq!(pattern.pattern_at(Point3::new(0.0, 0.0, 1.01)), BLACK);
    }
}
