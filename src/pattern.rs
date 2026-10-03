use crate::{
    canvas::Color,
    transforms::{Transform, TransformError},
};

use nalgebra::Point3;
use std::fmt::Debug;

const BLACK: Color = Color::new(0.0, 0.0, 0.0);
const WHITE: Color = Color::new(1.0, 1.0, 1.0);

pub trait Pattern: Debug {
    fn pattern_at(&self, world_point: Point3<f64>) -> Color;
    fn pattern_at_object(
        &self,
        object_transform: &Transform,
        world_point: Point3<f64>,
    ) -> Result<Color, TransformError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StripePattern {
    pub a: Color,
    pub b: Color,
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

#[derive(Debug)]
pub struct GradientPattern {
    pub a: Color,
    pub b: Color,
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

#[derive(Debug)]
pub struct RingPattern {
    pub a: Color,
    pub b: Color,
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

#[derive(Debug)]
pub struct CheckerPattern {
    pub a: Color,
    pub b: Color,
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
        if (world_point.x.floor() as i32
            + world_point.y.floor() as i32
            + world_point.z.floor() as i32)
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
