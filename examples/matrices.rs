use merlin_rt::transforms::{Axis, Transform};

use anyhow::Result;
use nalgebra::{Point3, Scale3, Translation3, Vector3};

fn main() -> Result<()> {
    // The identity transform leaves points and vectors unchanged.
    let point = Point3::new(1.0, 2.0, 3.0);
    let vector = Vector3::new(1.0, 2.0, 3.0);
    println!("identity * point: {}", Transform::identity().apply(point));
    println!("identity * vector: {}", Transform::identity().apply(vector));

    // Applying a transform's inverse restores the original point.
    let translation = Transform::translation(Translation3::new(5.0, -3.0, 2.0));
    let translated = translation.apply(point);
    let restored = translation.apply_inverse(translated)?;
    println!("translated point: {}", translated);
    println!("inverse(translation) * translated point: {}", restored);

    // Transforms can be composed and are applied in sequence order.
    let transform = Transform::sequence([
        Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2),
        Transform::scale(Scale3::new(2.0, 3.0, 4.0)),
        Transform::translation(Translation3::new(5.0, -3.0, 2.0)),
    ]);
    println!("composed transform * point: {}", transform.apply(point));

    // Scaling one axis changes the corresponding vector component.
    let scale_x = Transform::scale(Scale3::new(15.0, 1.0, 1.0));
    println!("scale_x * vector: {}", scale_x.apply(vector));

    Ok(())
}
