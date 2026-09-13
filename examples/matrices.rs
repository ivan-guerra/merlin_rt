use merlin_rt::primitives::Matrix;

use anyhow::Result;

fn main() -> Result<()> {
    // Answer to question 1: The inverse of the identity matrix is the identity matrix itself.
    let ident3 = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])?;
    let ident3_inv = ident3.inverse()?.expect("Matrix inversion failed");
    println!("ident3_inv:\n{}", ident3_inv);

    // Answer to question 2: The product of a matrix and its inverse is the identity matrix.
    let mat = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0])?;
    let mat_inv = mat.inverse()?.expect("Matrix inversion failed");
    let prod = (mat * mat_inv)?;
    println!("mat * mat_inv:\n{}", prod);

    // Answer to question 3: The transpose of the inverse of a matrix is equal to the inverse of the
    // transpose of that matrix.
    let mat = Matrix::from_vec(
        4,
        4,
        vec![
            6.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 6.0, 4.0, -9.0, 3.0, -7.0, 9.0, 1.0, 7.0, -6.0,
        ],
    )?;
    let mat_inv = mat.inverse()?.expect("Matrix inversion failed");
    let transpose = mat.transpose();
    let inv_trans = mat_inv.transpose();
    let trans_inv = transpose.inverse()?.expect("Matrix inversion failed");
    println!("inv_trans:\n{}", inv_trans);
    println!("trans_inv:\n{}", trans_inv);

    // Answer to question 4: Hard to describe see the question and the output lol.
    let vec = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0])?;
    let mod_ident3 = Matrix::from_vec(3, 3, vec![15.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])?;
    let prod = (mod_ident3 * vec)?;
    println!("mod_ident3 * vec:\n{}", prod);

    Ok(())
}
