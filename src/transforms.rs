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

#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    Translate(Translation3<f64>),
    Scale(Scale3<f64>),
    Reflection(Vector3<f64>),
    Rotate {
        axis: Axis,
        angle: f64,
    },
    Shear {
        xy: f64,
        xz: f64,
        yx: f64,
        yz: f64,
        zx: f64,
        zy: f64,
    },
    Identity,
    Sequence(Vec<Transform>),
}

impl Transform {
    pub fn sequence(transforms: impl IntoIterator<Item = Transform>) -> Self {
        let transforms: Vec<_> = transforms.into_iter().collect();

        match transforms.len() {
            0 => Self::Identity,
            1 => transforms.into_iter().next().unwrap(),
            _ => Self::Sequence(transforms),
        }
    }

    pub fn apply<T: Transformable>(&self, value: T) -> T {
        value.apply_transform(self)
    }

    pub fn apply_inverse<T: Transformable>(&self, value: T) -> Result<T, TransformError> {
        Ok(value.apply_matrix(&self.inverse()?))
    }

    pub fn apply_transpose_inverse<T: Transformable>(&self, value: T) -> Result<T, TransformError> {
        Ok(value.apply_matrix(&self.inverse()?.transpose()))
    }

    fn inverse(&self) -> Result<Matrix4<f64>, TransformError> {
        self.matrix()
            .try_inverse()
            .ok_or(TransformError::MatrixInversionError)
    }

    fn matrix(&self) -> Matrix4<f64> {
        match self {
            Self::Translate(translation) => translation.to_homogeneous(),
            Self::Scale(scale) => scale.to_homogeneous(),
            Self::Reflection(reflection) => {
                Translation3::new(reflection.x, reflection.y, reflection.z).to_homogeneous()
            }
            Self::Rotate { axis, angle } => match axis {
                Axis::X => Rotation3::from_axis_angle(&Vector3::x_axis(), *angle).to_homogeneous(),
                Axis::Y => Rotation3::from_axis_angle(&Vector3::y_axis(), *angle).to_homogeneous(),
                Axis::Z => Rotation3::from_axis_angle(&Vector3::z_axis(), *angle).to_homogeneous(),
            },
            Self::Shear {
                xy,
                xz,
                yx,
                yz,
                zx,
                zy,
            } => Matrix4::from_row_slice(&[
                1.0, *xy, *xz, 0.0, *yx, 1.0, *yz, 0.0, *zx, *zy, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
            Self::Sequence(transforms) => transforms
                .iter()
                .fold(Matrix4::identity(), |matrix, transform| {
                    transform.matrix() * matrix
                }),
            Self::Identity => Matrix4::identity(),
        }
    }
}

pub trait Transformable: Sized {
    fn apply_transform(self, transform: &Transform) -> Self;
    fn apply_matrix(self, matrix: &Matrix4<f64>) -> Self;
}

impl Transformable for Point3<f64> {
    fn apply_transform(self, transform: &Transform) -> Self {
        match transform {
            Transform::Translate(translation) => translation * self,
            Transform::Scale(scale) => scale * self,
            Transform::Reflection(_reflection) => todo!("Need to implement reflection for Point3"),
            Transform::Rotate { axis, angle } => match axis {
                Axis::X => Rotation3::from_axis_angle(&Vector3::x_axis(), *angle) * self,
                Axis::Y => Rotation3::from_axis_angle(&Vector3::y_axis(), *angle) * self,
                Axis::Z => Rotation3::from_axis_angle(&Vector3::z_axis(), *angle) * self,
            },
            Transform::Shear {
                xy,
                xz,
                yx,
                yz,
                zx,
                zy,
            } => Point3::new(
                self.x + xy * self.y + xz * self.z,
                self.y + yx * self.x + yz * self.z,
                self.z + zx * self.x + zy * self.y,
            ),
            Transform::Sequence(transforms) => transforms
                .iter()
                .fold(self, |point, transform| transform.apply(point)),
            Transform::Identity => self,
        }
    }

    fn apply_matrix(self, matrix: &Matrix4<f64>) -> Self {
        let point = matrix * Vector4::new(self.x, self.y, self.z, 1.0);
        Point3::new(point.x / point.w, point.y / point.w, point.z / point.w)
    }
}

impl Transformable for Vector3<f64> {
    fn apply_transform(self, transform: &Transform) -> Self {
        match transform {
            Transform::Translate(_) => self,
            Transform::Scale(scale) => scale * self,
            Transform::Reflection(reflection) => self - reflection * 2.0 * self.dot(reflection),
            Transform::Rotate { axis, angle } => match axis {
                Axis::X => Rotation3::from_axis_angle(&Vector3::x_axis(), *angle) * self,
                Axis::Y => Rotation3::from_axis_angle(&Vector3::y_axis(), *angle) * self,
                Axis::Z => Rotation3::from_axis_angle(&Vector3::z_axis(), *angle) * self,
            },
            Transform::Shear {
                xy,
                xz,
                yx,
                yz,
                zx,
                zy,
            } => Vector3::new(
                self.x + xy * self.y + xz * self.z,
                self.y + yx * self.x + yz * self.z,
                self.z + zx * self.x + zy * self.y,
            ),
            Transform::Sequence(transforms) => transforms
                .iter()
                .fold(self, |vector, transform| transform.apply(vector)),
            Transform::Identity => self,
        }
    }

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
        let transform = Transform::Translate(Translation3::new(5.0, -3.0, 2.0));
        let p = Point3::new(-3.0, 4.0, 5.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(2.0, 1.0, 7.0));
    }

    #[test]
    fn test_multiplying_by_the_inverse_of_a_translation_matrix() {
        let transform = Transform::Translate(Translation3::new(5.0, -3.0, 2.0));
        let p = Point3::new(-3.0, 4.0, 5.0);
        let p2 = transform.apply_inverse(p).unwrap();
        assert_eq!(p2, Point3::new(-8.0, 7.0, 3.0));
    }

    #[test]
    fn test_translation_does_not_affect_vectors() {
        let transform = Transform::Translate(Translation3::new(5.0, -3.0, 2.0));
        let v = Vector3::new(-3.0, 4.0, 5.0);
        let v2 = transform.apply(v);
        assert_eq!(v2, v);
    }

    #[test]
    fn test_scaling_matrix_applied_to_point() {
        let transform = Transform::Scale(Scale3::new(2.0, 3.0, 4.0));
        let p = Point3::new(-4.0, 6.0, 8.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(-8.0, 18.0, 32.0));
    }

    #[test]
    fn scaling_matrix_applied_to_vector() {
        let transform = Transform::Scale(Scale3::new(2.0, 3.0, 4.0));
        let v = Vector3::new(-4.0, 6.0, 8.0);
        let v2 = transform.apply(v);
        assert_eq!(v2, Vector3::new(-8.0, 18.0, 32.0));
    }

    #[test]
    fn test_multiplying_by_the_inverse_of_a_scaling_matrix() {
        let transform = Transform::Scale(Scale3::new(2.0, 3.0, 4.0));
        let v = Vector3::new(-4.0, 6.0, 8.0);
        let v2 = transform.apply_inverse(v).unwrap();
        assert_eq!(v2, Vector3::new(-2.0, 2.0, 2.0));
    }

    #[test]
    fn test_reflection_is_scaling_by_negative_value() {
        let transform = Transform::Scale(Scale3::new(-1.0, 1.0, 1.0));
        let p = Point3::new(2.0, 3.0, 4.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(-2.0, 3.0, 4.0));
    }

    #[test]
    fn test_rotating_a_point_around_x_axis() {
        let p = Point3::new(0.0, 1.0, 0.0);
        let half_quarter = Transform::Rotate {
            axis: Axis::X,
            angle: std::f64::consts::FRAC_PI_4,
        };
        let full_quarter = Transform::Rotate {
            axis: Axis::X,
            angle: std::f64::consts::FRAC_PI_2,
        };
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
        let half_quarter = Transform::Rotate {
            axis: Axis::X,
            angle: std::f64::consts::FRAC_PI_4,
        };
        let p2 = half_quarter.apply_inverse(p).unwrap();

        assert_abs_diff_eq!(p2.x, 0.0);
        assert_abs_diff_eq!(p2.y, 2f64.sqrt() / 2.0);
        assert_abs_diff_eq!(p2.z, -2f64.sqrt() / 2.0);
    }

    #[test]
    fn test_rotating_a_point_around_y_axis() {
        let p = Point3::new(0.0, 0.0, 1.0);
        let half_quarter = Transform::Rotate {
            axis: Axis::Y,
            angle: std::f64::consts::FRAC_PI_4,
        };
        let full_quarter = Transform::Rotate {
            axis: Axis::Y,
            angle: std::f64::consts::FRAC_PI_2,
        };
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
        let half_quarter = Transform::Rotate {
            axis: Axis::Z,
            angle: std::f64::consts::FRAC_PI_4,
        };
        let full_quarter = Transform::Rotate {
            axis: Axis::Z,
            angle: std::f64::consts::FRAC_PI_2,
        };
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
        let transform = Transform::Shear {
            xy: 1.0,
            xz: 0.0,
            yx: 0.0,
            yz: 0.0,
            zx: 0.0,
            zy: 0.0,
        };
        let p = Point3::new(2.0, 3.0, 4.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(5.0, 3.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_x_in_proportion_to_z() {
        let transform = Transform::Shear {
            xy: 0.0,
            xz: 1.0,
            yx: 0.0,
            yz: 0.0,
            zx: 0.0,
            zy: 0.0,
        };
        let p = Point3::new(2.0, 3.0, 4.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(6.0, 3.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_y_in_proportion_to_x() {
        let transform = Transform::Shear {
            xy: 0.0,
            xz: 0.0,
            yx: 1.0,
            yz: 0.0,
            zx: 0.0,
            zy: 0.0,
        };
        let p = Point3::new(2.0, 3.0, 4.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(2.0, 5.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_y_in_proportion_to_z() {
        let transform = Transform::Shear {
            xy: 0.0,
            xz: 0.0,
            yx: 0.0,
            yz: 1.0,
            zx: 0.0,
            zy: 0.0,
        };
        let p = Point3::new(2.0, 3.0, 4.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(2.0, 7.0, 4.0));
    }

    #[test]
    fn test_shearing_transformation_moves_z_in_proportion_to_x() {
        let transform = Transform::Shear {
            xy: 0.0,
            xz: 0.0,
            yx: 0.0,
            yz: 0.0,
            zx: 1.0,
            zy: 0.0,
        };
        let p = Point3::new(2.0, 3.0, 4.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(2.0, 3.0, 6.0));
    }

    #[test]
    fn test_shearing_transformation_moves_z_in_proportion_to_y() {
        let transform = Transform::Shear {
            xy: 0.0,
            xz: 0.0,
            yx: 0.0,
            yz: 0.0,
            zx: 0.0,
            zy: 1.0,
        };
        let p = Point3::new(2.0, 3.0, 4.0);
        let p2 = transform.apply(p);
        assert_eq!(p2, Point3::new(2.0, 3.0, 7.0));
    }

    #[test]
    fn test_individual_transformations_are_applied_in_sequence() {
        let p = Point3::new(1.0, 0.0, 1.0);
        let a = Transform::Rotate {
            axis: Axis::X,
            angle: std::f64::consts::FRAC_PI_2,
        };
        let b = Transform::Scale(Scale3::new(5.0, 5.0, 5.0));
        let c = Transform::Translate(Translation3::new(10.0, 5.0, 7.0));
        let p2 = a.apply(p);
        assert_abs_diff_eq!(p2.x, 1.0);
        assert_abs_diff_eq!(p2.y, -1.0);
        assert_abs_diff_eq!(p2.z, 0.0);

        let p3 = b.apply(p2);
        assert_abs_diff_eq!(p3.x, 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p3.y, -5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p3.z, 0.0, epsilon = 1e-12);

        let p4 = c.apply(p3);
        assert_abs_diff_eq!(p4.x, 15.0);
        assert_abs_diff_eq!(p4.y, 0.0);
        assert_abs_diff_eq!(p4.z, 7.0);
    }

    #[test]
    fn test_chained_transformations_must_be_applied_in_reverse_order() {
        let p = Point3::new(1.0, 0.0, 1.0);
        let a = Transform::Rotate {
            axis: Axis::X,
            angle: std::f64::consts::FRAC_PI_2,
        };
        let b = Transform::Scale(Scale3::new(5.0, 5.0, 5.0));
        let c = Transform::Translate(Translation3::new(10.0, 5.0, 7.0));
        let t = Transform::sequence(vec![a, b, c]);
        let p2 = t.apply(p);
        assert_abs_diff_eq!(p2.x, 15.0);
        assert_abs_diff_eq!(p2.y, 0.0);
        assert_abs_diff_eq!(p2.z, 7.0);
    }

    #[test]
    fn test_reflecting_a_vector_approaching_at_45_degrees() {
        let v = Vector3::new(1.0, -1.0, 0.0);
        let n = Vector3::new(0.0, 1.0, 0.0);
        let r = v.apply_transform(&Transform::Reflection(n));
        assert_abs_diff_eq!(r.x, 1.0);
        assert_abs_diff_eq!(r.y, 1.0);
        assert_abs_diff_eq!(r.z, 0.0);
    }

    #[test]
    fn test_reflecting_a_vector_off_a_slanted_surface() {
        let v = Vector3::new(0.0, -1.0, 0.0);
        let n = Vector3::new(2f64.sqrt() / 2.0, 2f64.sqrt() / 2.0, 0.0);
        let r = v.apply_transform(&Transform::Reflection(n));
        assert_abs_diff_eq!(r.x, 1.0);
        assert_abs_diff_eq!(r.y, 0.0);
        assert_abs_diff_eq!(r.z, 0.0);
    }
}
