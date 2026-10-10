use anyhow::{Context, Result, ensure};
use merlin_rt::{
    geometry::shapes::Group,
    rendering::{camera::Camera, canvas::Color},
    scene::{light::PointLight, world::World},
};
use nalgebra::{Point3, Vector3};
use std::{env, f64::consts::PI, path::PathBuf};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .context("Usage: cargo run --release --example wav -- <path/to/model.obj>")?,
    );
    ensure!(
        args.next().is_none(),
        "Expected exactly one argument: a path to an .obj file"
    );

    // Discard the input directory so the image is written in the working directory.
    let output = PathBuf::from(input.file_name().context("Input path must name a file")?)
        .with_extension("ppm");
    let model =
        Group::from_obj(&input).with_context(|| format!("Failed to load {}", input.display()))?;

    // Keep the group alive so its triangles can resolve their parent links.
    let world = World::from_shared(
        PointLight::new(Point3::new(-5.0, 6.0, -8.0), Color::new(1.0, 1.0, 1.0)),
        vec![model],
    );
    // This view assumes a model near the origin; adjust it for other model scales.
    let camera = Camera::new(
        400,
        300,
        PI / 3.0,
        World::view_transform(
            Point3::new(12.0, 9.0, -18.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    camera
        .render(&world)
        .context("Failed to render the model")?
        .write_to_ppm(&output)
        .with_context(|| format!("Failed to write {}", output.display()))?;
    println!("Wrote {}", output.display());
    Ok(())
}
