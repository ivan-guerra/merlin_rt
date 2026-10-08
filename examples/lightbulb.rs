use anyhow::Result;
use merlin_rt::{
    camera::Camera,
    canvas::Color,
    light::PointLight,
    material::Material,
    scene::World,
    shapes::{Cylinder, DoubleNappedCone, Plane, Shape, Sphere},
    transforms::Transform,
};
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    let bulb = Sphere {
        material: Material {
            color: Color::new(1.0, 0.75, 0.25),
            ambient: 0.2,
            diffuse: 0.35,
            specular: 0.9,
            shininess: 250.0,
            reflective: 0.15,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(1.15, 1.35, 1.15)),
            Transform::translation(Translation3::new(0.0, 1.25, 0.0)),
        ]),
        ..Default::default()
    };

    let neck = Cylinder {
        minimum: -0.5,
        maximum: 0.5,
        closed: true,
        material: Material {
            color: Color::new(1.0, 0.75, 0.25),
            ambient: 0.2,
            diffuse: 0.35,
            specular: 0.9,
            shininess: 250.0,
            reflective: 0.15,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.55, 0.75, 0.55)),
            Transform::translation(Translation3::new(0.0, -0.05, 0.0)),
        ]),
    };

    let socket = DoubleNappedCone {
        minimum: -0.45,
        maximum: 0.45,
        closed: true,
        material: Material {
            color: Color::new(0.3, 0.32, 0.35),
            ambient: 0.1,
            diffuse: 0.55,
            specular: 0.8,
            shininess: 150.0,
            reflective: 0.35,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.9, 0.75, 0.9)),
            Transform::translation(Translation3::new(0.0, -0.85, 0.0)),
        ]),
    };

    let base = Cylinder {
        minimum: -0.35,
        maximum: 0.35,
        closed: true,
        material: Material {
            color: Color::new(0.08, 0.09, 0.12),
            ambient: 0.1,
            diffuse: 0.5,
            specular: 0.7,
            shininess: 100.0,
            reflective: 0.25,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.65, 0.55, 0.65)),
            Transform::translation(Translation3::new(0.0, -1.55, 0.0)),
        ]),
    };

    let floor = Plane {
        material: Material {
            color: Color::new(1.0, 1.0, 1.0),
            ambient: 0.15,
            diffuse: 0.7,
            specular: 0.35,
            shininess: 75.0,
            reflective: 0.15,
            ..Default::default()
        },
        transform: Transform::translation(Translation3::new(0.0, -1.95, 0.0)),
    };

    let mut objects: Vec<Box<dyn Shape>> = vec![
        Box::new(floor),
        Box::new(bulb),
        Box::new(neck),
        Box::new(socket),
        Box::new(base),
    ];

    for y in [-1.15, -1.4, -1.65] {
        objects.push(Box::new(Cylinder {
            minimum: -0.1,
            maximum: 0.1,
            closed: true,
            material: Material {
                color: Color::new(0.3, 0.32, 0.35),
                ambient: 0.1,
                diffuse: 0.55,
                specular: 0.8,
                shininess: 150.0,
                reflective: 0.35,
                ..Default::default()
            },
            transform: Transform::sequence([
                Transform::scale(Scale3::new(0.78, 0.18, 0.78)),
                Transform::translation(Translation3::new(0.0, y, 0.0)),
            ]),
        }));
    }

    let world = World::new(
        PointLight::new(Point3::new(-5.0, 7.0, -6.0), Color::new(1.0, 1.0, 1.0)),
        objects,
    );

    let camera = Camera::new(
        400,
        300,
        std::f64::consts::PI / 3.0,
        World::view_transform(
            Point3::new(4.5, 2.5, -7.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("lightbulb.ppm"))?;

    Ok(())
}
