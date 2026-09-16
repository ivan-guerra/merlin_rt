use nalgebra::{Matrix4, Point3, Rotation3, Scale3, Translation3, Vector3, Vector4};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransformError {
    #[error("Failed to invert transformation matrix")]
    MatrixInversionError,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    matrix: Matrix4<f64>,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            matrix: Matrix4::identity(),
        }
    }

    pub fn translation(translation: Translation3<f64>) -> Self {
        Self {
            matrix: translation.to_homogeneous(),
        }
    }

    pub fn scale(scale: Scale3<f64>) -> Self {
        Self {
            matrix: scale.to_homogeneous(),
        }
    }

    pub fn reflection(normal: Vector3<f64>) -> Self {
        let normal = normal.normalize();

        Self {
            matrix: Matrix4::from_row_slice(&[
                1.0 - 2.0 * normal.x * normal.x,
                -2.0 * normal.x * normal.y,
                -2.0 * normal.x * normal.z,
                0.0,
                -2.0 * normal.y * normal.x,
                1.0 - 2.0 * normal.y * normal.y,
                -2.0 * normal.y * normal.z,
                0.0,
                -2.0 * normal.z * normal.x,
                -2.0 * normal.z * normal.y,
                1.0 - 2.0 * normal.z * normal.z,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]),
        }
    }

    pub fn rotation(axis: Axis, angle: f64) -> Self {
        let rotation = match axis {
            Axis::X => Rotation3::from_axis_angle(&Vector3::x_axis(), angle),
            Axis::Y => Rotation3::from_axis_angle(&Vector3::y_axis(), angle),
            Axis::Z => Rotation3::from_axis_angle(&Vector3::z_axis(), angle),
        };

        Self {
            matrix: rotation.to_homogeneous(),
        }
    }

    pub fn shear(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Self {
        Self {
            matrix: Matrix4::from_row_slice(&[
                1.0, xy, xz, 0.0, yx, 1.0, yz, 0.0, zx, zy, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
        }
    }

    pub fn sequence(transforms: impl IntoIterator<Item = Self>) -> Self {
        transforms
            .into_iter()
            .fold(Self::identity(), |matrix, transform| Self {
                matrix: transform.matrix * matrix.matrix,
            })
    }

    pub fn apply<T: Transformable>(&self, value: T) -> T {
        value.apply_matrix(&self.matrix)
    }

    pub fn apply_inverse<T: Transformable>(&self, value: T) -> Result<T, TransformError> {
        Ok(value.apply_matrix(&self.inverse()?))
    }

    pub fn apply_transpose_inverse<T: Transformable>(&self, value: T) -> Result<T, TransformError> {
        Ok(value.apply_matrix(&self.inverse()?.transpose()))
    }

    fn inverse(&self) -> Result<Matrix4<f64>, TransformError> {
        self.matrix
            .try_inverse()
            .ok_or(TransformError::MatrixInversionError)
    }
}

pub trait Transformable: Sized {
    fn apply_matrix(self, matrix: &Matrix4<f64>) -> Self;
}

impl Transformable for Point3<f64> {
    fn apply_matrix(self, matrix: &Matrix4<f64>) -> Self {
        let point = matrix * Vector4::new(self.x, self.y, self.z, 1.0);
        Point3::new(point.x / point.w, point.y / point.w, point.z / point.w)
    }
}

impl Transformable for Vector3<f64> {
    fn apply_matrix(self, matrix: &Matrix4<f64>) -> Self {
        let vector = matrix * Vector4::new(self.x, self.y, self.z, 0.0);
        Vector3::new(vector.x, vector.y, vector.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_multiplying_by_a_translation_matrix() {
        let transform = Transform::translation(Translation3::new(5.0, -3.0, 2.0));
        let p = Point3::new(-3.0, 4.0, 5.0);

        assert_eq!(transform.apply(p), Point3::new(2.0, 1.0, 7.0));
    }

    #[test]
    fn test_multiplying_by_the_inverse_of_a_translation_matrix() {
        let transform = Transform::translation(Translation3::new(5.0, -3.0, 2.0));
        let p = Point3::new(-3.0, 4.0, 5.0);

        assert_eq!(
            transform.apply_inverse(p).unwrap(),
            Point3::new(-8.0, 7.0, 3.0)
        );
    }

    #[test]
    fn test_translation_does_not_affect_vectors() {
        let transform = Transform::translation(Translation3::new(5.0, -3.0, 2.0));
        let v = Vector3::new(-3.0, 4.0, 5.0);

        assert_eq!(transform.apply(v), v);
    }

    #[test]
    fn test_scaling_matrix_applied_to_point() {
        let transform = Transform::scale(Scale3::new(2.0, 3.0, 4.0));
        let p = Point3::new(-4.0, 6.0, 8.0);

        assert_eq!(transform.apply(p), Point3::new(-8.0, 18.0, 32.0));
    }

    #[test]
    fn test_scaling_matrix_applied_to_vector() {
        let transform = Transform::scale(Scale3::new(2.0, 3.0, 4.0));
        let v = Vector3::new(-4.0, 6.0, 8.0);

        assert_eq!(transform.apply(v), Vector3::new(-8.0, 18.0, 32.0));
    }

    #[test]
    fn test_multiplying_by_the_inverse_of_a_scaling_matrix() {
        let transform = Transform::scale(Scale3::new(2.0, 3.0, 4.0));
        let v = Vector3::new(-4.0, 6.0, 8.0);

        assert_eq!(
            transform.apply_inverse(v).unwrap(),
            Vector3::new(-2.0, 2.0, 2.0)
        );
    }

    #[test]
    fn test_reflection_is_scaling_by_negative_value() {
        let transform = Transform::scale(Scale3::new(-1.0, 1.0, 1.0));
        let p = Point3::new(2.0, 3.0, 4.0);

        assert_eq!(transform.apply(p), Point3::new(-2.0, 3.0, 4.0));
    }

    #[test]
    fn test_rotating_a_point_around_x_axis() {
        let p = Point3::new(0.0, 1.0, 0.0);
        let half_quarter = Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_4);
        let full_quarter = Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2);

        let p2 = half_quarter.apply(p);
        let p3 = full_quarter.apply(p);

        assert_abs_diff_eq!(p2.x, 0.0);
        assert_abs_diff_eq!(p2.y, 2f64.sqrt() / 2.0);
        assert_abs_diff_eq!(p2.z, 2f64.sqrt() / 2.0);

        assert_abs_diff_eq!(p3.x, 0.0);
        assert_abs_diff_eq!(p3.y, 0.0);
        assert_abs_diff_eq!(p3.z, 1.0);
    }

    #[test]
    fn test_inverse_of_an_x_rotation_rotates_in_opposite_direction() {
        let p = Point3::new(0.0, 1.0, 0.0);
        let half_quarter = Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_4);
        let p2 = half_quarter.apply_inverse(p).unwrap();

        assert_abs_diff_eq!(p2.x, 0.0);
        assert_abs_diff_eq!(p2.y, 2f64.sqrt() / 2.0);
        assert_abs_diff_eq!(p2.z, -2f64.sqrt() / 2.0);
    }

    #[test]
    fn test_rotating_a_point_around_y_axis() {
        let p = Point3::new(0.0, 0.0, 1.0);
        let half_quarter = Transform::rotation(Axis::Y, std::f64::consts::FRAC_PI_4);
        let full_quarter = Transform::rotation(Axis::Y, std::f64::consts::FRAC_PI_2);

        let p2 = half_quarter.apply(p);
        let p3 = full_quarter.apply(p);

        assert_abs_diff_eq!(p2.x, 2f64.sqrt() / 2.0);
        assert_abs_diff_eq!(p2.y, 0.0);
        assert_abs_diff_eq!(p2.z, 2f64.sqrt() / 2.0);

        assert_abs_diff_eq!(p3.x, 1.0);
        assert_abs_diff_eq!(p3.y, 0.0);
        assert_abs_diff_eq!(p3.z, 0.0);
    }

    #[test]
    fn test_rotating_a_point_around_z_axis() {
        let p = Point3::new(0.0, 1.0, 0.0);
        let half_quarter = Transform::rotation(Axis::Z, std::f64::consts::FRAC_PI_4);
        let full_quarter = Transform::rotation(Axis::Z, std::f64::consts::FRAC_PI_2);

        let p2 = half_quarter.apply(p);
        let p3 = full_quarter.apply(p);

        assert_abs_diff_eq!(p2.x, -2f64.sqrt() / 2.0);
        assert_abs_diff_eq!(p2.y, 2f64.sqrt() / 2.0);
        assert_abs_diff_eq!(p2.z, 0.0);

        assert_abs_diff_eq!(p3.x, -1.0);
        assert_abs_diff_eq!(p3.y, 0.0);
        assert_abs_diff_eq!(p3.z, 0.0);
    }

    #[test]
    fn test_shearing_transformation_moves_x_in_proportion_to_y() {
        let transform = Transform::shear(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let p = Point3::new(2.0, 3.0, 4.0);

        assert_eq!(transform.apply(p), Point3::new(5.0, 3.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_x_in_proportion_to_z() {
        let transform = Transform::shear(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let p = Point3::new(2.0, 3.0, 4.0);

        assert_eq!(transform.apply(p), Point3::new(6.0, 3.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_y_in_proportion_to_x() {
        let transform = Transform::shear(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let p = Point3::new(2.0, 3.0, 4.0);

        assert_eq!(transform.apply(p), Point3::new(2.0, 5.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_y_in_proportion_to_z() {
        let transform = Transform::shear(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let p = Point3::new(2.0, 3.0, 4.0);

        assert_eq!(transform.apply(p), Point3::new(2.0, 7.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_z_in_proportion_to_x() {
        let transform = Transform::shear(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let p = Point3::new(2.0, 3.0, 4.0);

        assert_eq!(transform.apply(p), Point3::new(2.0, 3.0, 6.0));
    }

    #[test]
    fn test_shearing_transformation_moves_z_in_proportion_to_y() {
        let transform = Transform::shear(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let p = Point3::new(2.0, 3.0, 4.0);

        assert_eq!(transform.apply(p), Point3::new(2.0, 3.0, 7.0));
    }

    #[test]
    fn test_individual_transformations_are_applied_in_sequence() {
        let p = Point3::new(1.0, 0.0, 1.0);
        let a = Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2);
        let b = Transform::scale(Scale3::new(5.0, 5.0, 5.0));
        let c = Transform::translation(Translation3::new(10.0, 5.0, 7.0));

        let p2 = a.apply(p);
        let p3 = b.apply(p2);
        let p4 = c.apply(p3);

        assert_abs_diff_eq!(p2, Point3::new(1.0, -1.0, 0.0), epsilon = 1e-10);
        assert_abs_diff_eq!(p3, Point3::new(5.0, -5.0, 0.0), epsilon = 1e-10);
        assert_abs_diff_eq!(p4, Point3::new(15.0, 0.0, 7.0), epsilon = 1e-10);
    }

    #[test]
    fn test_chained_transformations_must_be_applied_in_reverse_order() {
        let p = Point3::new(1.0, 0.0, 1.0);
        let transform = Transform::sequence([
            Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2),
            Transform::scale(Scale3::new(5.0, 5.0, 5.0)),
            Transform::translation(Translation3::new(10.0, 5.0, 7.0)),
        ]);

        assert_abs_diff_eq!(transform.apply(p), Point3::new(15.0, 0.0, 7.0));
    }

    #[test]
    fn test_reflecting_a_vector_approaching_at_45_degrees() {
        let v = Vector3::new(1.0, -1.0, 0.0);
        let normal = Vector3::new(0.0, 1.0, 0.0);

        assert_abs_diff_eq!(
            Transform::reflection(normal).apply(v),
            Vector3::new(1.0, 1.0, 0.0)
        );
    }

    #[test]
    fn test_reflecting_a_vector_off_a_slanted_surface() {
        let v = Vector3::new(0.0, -1.0, 0.0);
        let normal = Vector3::new(2f64.sqrt() / 2.0, 2f64.sqrt() / 2.0, 0.0);

        assert_abs_diff_eq!(
            Transform::reflection(normal).apply(v),
            Vector3::new(1.0, 0.0, 0.0)
        );
    }
}
