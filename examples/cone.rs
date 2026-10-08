use anyhow::Result;
use merlin_rt::{
    geometry::{shapes::DoubleNappedCone, transforms::Transform},
    rendering::{camera::Camera, canvas::Color},
    scene::{light::PointLight, material::Material, world::World},
};
use nalgebra::{Point3, Scale3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    let cone = DoubleNappedCone {
        minimum: -1.5,
        maximum: 1.5,
        closed: true,
        material: Material {
            color: Color::new(0.85, 0.35, 0.15),
            diffuse: 0.7,
            specular: 0.3,
            shininess: 100.0,
            ..Default::default()
        },
        transform: Transform::scale(Scale3::new(1.0, 1.5, 1.0)),
    };

    let world = World::new(
        PointLight::new(Point3::new(-5.0, 6.0, -8.0), Color::new(1.0, 1.0, 1.0)),
        vec![Box::new(cone)],
    );

    let camera = Camera::new(
        400,
        300,
        std::f64::consts::PI / 3.0,
        World::view_transform(
            Point3::new(4.0, 3.0, -6.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("cone.ppm"))?;

    Ok(())
}
