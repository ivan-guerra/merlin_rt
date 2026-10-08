use merlin_rt::rendering::canvas::{Canvas, Color};

use anyhow::Result;
use nalgebra::{Point3, Rotation3, Scale3, Translation3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    const SCALE_FACTOR: f64 = 3.0 / 8.0;
    const HOURS: usize = 12;

    let white = Color::new(1.0, 1.0, 1.0);
    let mut canvas = Canvas::new(512, 512);
    let center_trans =
        Translation3::new(canvas.width as f64 / 2.0, canvas.height as f64 / 2.0, 0.0);
    let radius = canvas.width.min(canvas.height) as f64 * SCALE_FACTOR;
    let scale = Scale3::new(radius, radius, 1.0);
    let origin = Point3::new(0.0, 1.0, 0.0);

    for hour in 1..=HOURS {
        let angle = -(hour as f64) * std::f64::consts::FRAC_PI_6;
        let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), angle);

        let point = center_trans * (scale * (rotation * origin));
        canvas.write_pixel(point.x as usize, point.y as usize, white)?;
    }

    canvas.write_to_ppm(Path::new("clock.ppm"))?;

    Ok(())
}
