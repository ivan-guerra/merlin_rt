//! Render the cover scene from The Ray Tracer Challenge.
//!
//! Run with `cargo run --release --example cover` to write `cover.ppm` (100×100).

use anyhow::Result;
use merlin_rt::{
    geometry::{
        shapes::{Cube, Plane, Shape, Sphere},
        transforms::{Axis, Transform},
    },
    rendering::{camera::Camera, canvas::Color},
    scene::{light::PointLight, material::Material, world::World},
};
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::{f64::consts::FRAC_PI_2, path::Path};

fn surface(color: Color) -> Material {
    Material {
        color,
        diffuse: 0.7,
        ambient: 0.1,
        specular: 0.0,
        reflective: 0.1,
        ..Default::default()
    }
}

fn object_transform(size: f64) -> Transform {
    // YAML order: standard-transform (translate, then scale), then size scale.
    Transform::sequence([
        Transform::translation(Translation3::new(1.0, -1.0, 1.0)),
        Transform::scale(Scale3::new(0.5, 0.5, 0.5)),
        Transform::scale(Scale3::new(size, size, size)),
    ])
}

fn main() -> Result<()> {
    let white = Color::new(1.0, 1.0, 1.0);
    let blue = Color::new(0.537, 0.831, 0.914);
    let red = Color::new(0.941, 0.322, 0.388);
    let purple = Color::new(0.373, 0.404, 0.550);

    let large = object_transform(3.5);
    let medium = object_transform(3.0);
    let small = object_transform(2.0);

    let backdrop = Plane::builder()
        .material(Material {
            color: white,
            ambient: 1.0,
            diffuse: 0.0,
            specular: 0.0,
            ..Default::default()
        })
        .transform(Transform::sequence([
            Transform::rotation(Axis::X, FRAC_PI_2),
            Transform::translation(Translation3::new(0.0, 0.0, 500.0)),
        ]))
        .build();

    let sphere = Sphere::builder()
        .material(Material {
            color: purple,
            diffuse: 0.2,
            ambient: 0.0,
            specular: 1.0,
            shininess: 200.0,
            reflective: 0.7,
            transparency: 0.7,
            refractive_index: 1.5,
            ..Default::default()
        })
        .transform(large)
        .build();

    // Color, named size transform, and final translation, in YAML order.
    let cubes = [
        (white, medium, [4.0, 0.0, 0.0]),
        (blue, large, [8.5, 1.5, -0.5]),
        (red, large, [0.0, 0.0, 4.0]),
        (white, small, [4.0, 0.0, 4.0]),
        (purple, medium, [7.5, 0.5, 4.0]),
        (white, medium, [-0.25, 0.25, 8.0]),
        (blue, large, [4.0, 1.0, 7.5]),
        (red, medium, [10.0, 2.0, 7.5]),
        (white, small, [8.0, 2.0, 12.0]),
        (white, small, [20.0, 1.0, 9.0]),
        (blue, large, [-0.5, -5.0, 0.25]),
        (red, large, [4.0, -4.0, 0.0]),
        (white, large, [8.5, -4.0, 0.0]),
        (white, large, [0.0, -4.0, 4.0]),
        (purple, large, [-0.5, -4.5, 8.0]),
        (white, large, [0.0, -8.0, 4.0]),
        (white, large, [-0.5, -8.5, 8.0]),
    ];

    let mut objects: Vec<Box<dyn Shape>> = vec![Box::new(backdrop), Box::new(sphere)];
    for (color, size, [x, y, z]) in cubes {
        objects.push(Box::new(
            Cube::builder()
                .material(surface(color))
                .transform(Transform::sequence([
                    size,
                    Transform::translation(Translation3::new(x, y, z)),
                ]))
                .build(),
        ));
    }

    let mut world = World::new(
        PointLight::new(Point3::new(50.0, 100.0, -50.0), white),
        objects,
    );
    world.add_light(PointLight::new(
        Point3::new(-400.0, 50.0, -10.0),
        Color::new(0.2, 0.2, 0.2),
    ));

    let camera = Camera::new(
        500,
        500,
        0.65,
        World::view_transform(
            Point3::new(-6.0, 6.0, -10.0),
            Point3::new(6.0, 0.0, 6.0),
            Vector3::new(-0.45, 1.0, 0.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("cover.ppm"))?;

    Ok(())
}
