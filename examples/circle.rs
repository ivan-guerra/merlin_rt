use merlin_rt::canvas::{Canvas, Color};
use merlin_rt::material::Material;
use merlin_rt::ray::{Intersectable, Intersection, Ray};
use merlin_rt::sphere::Sphere;
use merlin_rt::transforms::{Axis, Transform};

use anyhow::Result;
use nalgebra::{Point3, Scale3};
use std::path::Path;

fn main() -> Result<()> {
    const CANVAS_PIXELS: usize = 512;

    let intersect_color = Color::new(1.0, 0.0, 0.0);
    let mut canvas = Canvas::new(CANVAS_PIXELS, CANVAS_PIXELS);
    let sphere = Sphere::new(
        1.0,
        Point3::new(0.0, 0.0, 0.0),
        Transform::sequence([
            Transform::Scale(Scale3::new(0.5, 1.0, 1.0)),
            Transform::Rotate {
                axis: Axis::Z,
                angle: std::f64::consts::FRAC_PI_4,
            },
        ]),
        Material::default(),
    );
    let wall_size = 7.0;
    let wall_z = 10.0;
    let pixel_size = wall_size / CANVAS_PIXELS as f64;
    let half = wall_size / 2.0;
    let ray_origin = Point3::new(0.0, 0.0, -5.0);

    for y in 0..CANVAS_PIXELS {
        let world_y = half - pixel_size * y as f64;
        for x in 0..CANVAS_PIXELS {
            let world_x = -half + pixel_size * x as f64;
            let position = Point3::new(world_x, world_y, wall_z);
            let ray = Ray::new(ray_origin, (position - ray_origin).normalize());
            let mut intersections = sphere.intersect(&ray)?;

            if Intersection::hit(&mut intersections).is_some() {
                canvas.write_pixel(x, y, intersect_color)?;
            }
        }
    }

    canvas.write_to_ppm(Path::new("circle.ppm"))?;

    Ok(())
}
