use merlin_rt::{
    geometry::{
        shapes::Sphere,
        transforms::{Axis, Transform},
    },
    rendering::{camera::Camera, canvas::Color},
    scene::{light::PointLight, material::Material, world::World},
};

use anyhow::Result;
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    let floor = Sphere::builder()
        .transform(Transform::scale(Scale3::new(10.0, 0.01, 10.0)))
        .material(Material {
            color: Color::new(1.0, 0.9, 0.9),
            specular: 0.0,
            ..Default::default()
        })
        .build();
    let left_wall = Sphere::builder()
        .transform(Transform::sequence([
            Transform::scale(Scale3::new(10.0, 0.01, 10.0)),
            Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2),
            Transform::rotation(Axis::Y, -std::f64::consts::FRAC_PI_4),
            Transform::translation(Translation3::new(0.0, 0.0, 5.0)),
        ]))
        .material(Material {
            color: Color::new(1.0, 0.9, 0.9),
            specular: 0.0,
            ..Default::default()
        })
        .build();
    let right_wall = Sphere::builder()
        .transform(Transform::sequence([
            Transform::scale(Scale3::new(10.0, 0.01, 10.0)),
            Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2),
            Transform::rotation(Axis::Y, std::f64::consts::FRAC_PI_4),
            Transform::translation(Translation3::new(0.0, 0.0, 5.0)),
        ]))
        .material(Material {
            color: Color::new(1.0, 0.9, 0.9),
            specular: 0.0,
            ..Default::default()
        })
        .build();
    let middle = Sphere::builder()
        .transform(Transform::translation(Translation3::new(-0.5, 1.0, 0.5)))
        .material(Material {
            color: Color::new(0.1, 1.0, 0.5),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        })
        .build();
    let right = Sphere::builder()
        .transform(Transform::sequence([
            Transform::scale(Scale3::new(0.5, 0.5, 0.5)),
            Transform::translation(Translation3::new(1.5, 0.5, -0.5)),
        ]))
        .material(Material {
            color: Color::new(0.5, 1.0, 0.1),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        })
        .build();
    let left = Sphere::builder()
        .transform(Transform::sequence([
            Transform::scale(Scale3::new(0.33, 0.33, 0.33)),
            Transform::translation(Translation3::new(-1.5, 0.33, -0.75)),
        ]))
        .material(Material {
            color: Color::new(1.0, 0.8, 0.1),
            diffuse: 0.7,
            specular: 0.3,
            ..Default::default()
        })
        .build();
    let world = World::new(
        PointLight::new(Point3::new(-10.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0)),
        vec![
            Box::new(floor),
            Box::new(left_wall),
            Box::new(right_wall),
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

    canvas.write_to_ppm(Path::new("scene.ppm"))?;

    Ok(())
}
