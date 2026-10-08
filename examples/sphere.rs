use merlin_rt::{
    geometry::{
        ray::Ray,
        shapes::{Intersection, Shape, Sphere},
        transforms::Transform,
    },
    rendering::canvas::{Canvas, Color},
    scene::{light::PointLight, material::Material},
};

use anyhow::Result;
use nalgebra::Point3;
use std::path::Path;

fn main() -> Result<()> {
    const CANVAS_PIXELS: usize = 512;
    const WALL_SIZE: f64 = 7.0;
    const WALL_Z: f64 = 10.0;

    let mut canvas = Canvas::new(CANVAS_PIXELS, CANVAS_PIXELS);
    let sphere = Sphere::new(
        1.0,
        Point3::origin(),
        Transform::identity(),
        Material {
            color: Color::new(1.0, 0.2, 1.0),
            ..Default::default()
        },
    );
    let light = PointLight::new(Point3::new(-10.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0));
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

            let Some(hit) = Intersection::hit(&mut intersections) else {
                continue;
            };

            let point = ray.position(hit.t);
            let normal = hit.object.normal_at(point)?;
            let color = hit
                .object
                .lighting(light, point, -direction, normal, false)?;
            canvas.write_pixel(x, y, color)?;
        }
    }

    canvas.write_to_ppm(Path::new("sphere.ppm"))?;
    Ok(())
}
