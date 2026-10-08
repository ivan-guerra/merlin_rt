use merlin_rt::{
    camera::Camera,
    canvas::Color,
    light::PointLight,
    material::Material,
    scene::World,
    shapes::{Plane, Sphere},
    transforms::Transform,
};

use anyhow::Result;
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    let floor = Plane::default();

    let middle = Sphere {
        transform: Transform::translation(Translation3::new(-0.5, 1.0, 0.5)),
        material: Material {
            color: Color::new(0.1, 1.0, 0.5),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        },
        ..Default::default()
    };

    let right = Sphere {
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.5, 0.5, 0.5)),
            Transform::translation(Translation3::new(1.5, 0.5, -0.5)),
        ]),
        material: Material {
            color: Color::new(0.5, 1.0, 0.1),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        },
        ..Default::default()
    };

    let left = Sphere {
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.33, 0.33, 0.33)),
            Transform::translation(Translation3::new(-1.5, 0.33, -0.75)),
        ]),
        material: Material {
            color: Color::new(1.0, 0.8, 0.1),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        },
        ..Default::default()
    };

    let world = World::new(
        PointLight::new(Point3::new(-10.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0)),
        vec![
            Box::new(floor),
            Box::new(middle),
            Box::new(right),
            Box::new(left),
        ],
    );

    let camera = Camera::new(
        1000,
        500,
        std::f64::consts::PI / 3.0,
        World::view_transform(
            Point3::new(0.0, 1.5, -5.0),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("plane.ppm"))?;

    Ok(())
}
