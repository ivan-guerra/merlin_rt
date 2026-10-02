use merlin_rt::{
    canvas::{Canvas, Color},
    material::Material,
    ray::{Ray},
    shape::{Intersection, Shape},
    sphere::Sphere,
    transforms::{Axis, Transform},
};

use anyhow::Result;
use nalgebra::{Point3, Scale3};
use std::path::Path;

fn main() -> Result<()> {
    const CANVAS_PIXELS: usize = 512;
    const WALL_SIZE: f64 = 7.0;
    const WALL_Z: f64 = 10.0;

    let mut canvas = Canvas::new(CANVAS_PIXELS, CANVAS_PIXELS);
    let sphere = Sphere::new(
        1.0,
        Point3::origin(),
        Transform::sequence([
            Transform::scale(Scale3::new(0.5, 1.0, 1.0)),
            Transform::rotation(Axis::Z, std::f64::consts::FRAC_PI_4),
        ]),
        Material::default(),
    );
    let hit_color = Color::new(1.0, 0.0, 0.0);
    let ray_origin = Point3::new(0.0, 0.0, -5.0);
    let pixel_size = WALL_SIZE / CANVAS_PIXELS as f64;
    let half = WALL_SIZE / 2.0;

    for y in 0..CANVAS_PIXELS {
        let world_y = half - y as f64 * pixel_size;

        for x in 0..CANVAS_PIXELS {
            let world_x = -half + x as f64 * pixel_size;
            let position = Point3::new(world_x, world_y, WALL_Z);
            let direction = (position - ray_origin).normalize();
            let ray = Ray::new(ray_origin, direction);
            let mut intersections = sphere.intersect(&ray)?;

            let Some(_) = Intersection::hit(&mut intersections) else {
                continue;
            };

            canvas.write_pixel(x, y, hit_color)?;
        }
    }

    canvas.write_to_ppm(Path::new("circle.ppm"))?;
    Ok(())
}
