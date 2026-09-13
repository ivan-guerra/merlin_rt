use anyhow::Result;
use nalgebra::{Matrix3, Matrix4, Vector3};

fn main() -> Result<()> {
    // Answer to question 1: The inverse of the identity matrix is the identity matrix itself.
    let ident3 = Matrix3::<f64>::identity();
    let ident3_inv = ident3.try_inverse().expect("matrix inversion failed");
    println!("ident3_inv:\n{}", ident3_inv);

    // Answer to question 2: The product of a matrix and its inverse is the identity matrix.
    let mat = Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);
    let mat_inv = mat.try_inverse().expect("matrix inversion failed");
    let prod = mat * mat_inv;
    println!("mat * mat_inv:\n{}", prod);

    // Answer to question 3: The transpose of the inverse of a matrix is equal to the inverse of the
    // transpose of that matrix.
    let mat = Matrix4::new(
        6.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 6.0, 4.0, -9.0, 3.0, -7.0, 9.0, 1.0, 7.0, -6.0,
    );
    let mat_inv = mat.try_inverse().expect("matrix inversion failed");
    let inv_trans = mat_inv.transpose();
    let trans_inv = mat
        .transpose()
        .try_inverse()
        .expect("matrix inversion failed");
    println!("inv_trans:\n{}", inv_trans);
    println!("trans_inv:\n{}", trans_inv);

    // Answer to question 4: Scaling one axis changes the corresponding vector component.
    let vector = Vector3::new(1.0, 2.0, 3.0);
    let mod_ident3 = Matrix3::new(15.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let prod = mod_ident3 * vector;
    println!("mod_ident3 * vector:\n{}", prod);

    Ok(())
}
