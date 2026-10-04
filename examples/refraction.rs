use anyhow::Result;
use merlin_rt::{
    camera::Camera, canvas::Color, light::PointLight, material::Material, pattern::CheckerPattern,
    plane::Plane, scene::World, sphere::Sphere, transforms::Transform,
};
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    let floor = Plane {
        material: Material {
            ambient: 0.15,
            diffuse: 0.75,
            specular: 0.15,
            reflective: 0.25,
            pattern: Some(Box::new(CheckerPattern {
                a: Color::new(0.9, 0.9, 0.95),
                b: Color::new(0.08, 0.1, 0.16),
                transform: Transform::scale(Scale3::new(0.5, 0.5, 0.5)),
            })),
            ..Default::default()
        },
        transform: Transform::identity(),
    };

    let glass_sphere = Sphere {
        material: Material {
            color: Color::new(0.95, 0.98, 1.0),
            ambient: 0.0,
            diffuse: 0.1,
            specular: 0.9,
            shininess: 300.0,
            reflective: 1.0,
            transparency: 1.0,
            refractive_index: 1.5,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(1.5, 1.5, 1.5)),
            Transform::translation(Translation3::new(0.0, 1.5, 0.0)),
        ]),
        ..Default::default()
    };

    let air_bubble = Sphere {
        material: Material {
            ambient: 0.0,
            diffuse: 0.0,
            specular: 0.0,
            reflective: 1.0,
            transparency: 1.0,
            refractive_index: 1.0,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.45, 0.45, 0.45)),
            Transform::translation(Translation3::new(0.0, 1.5, 0.0)),
        ]),
        ..Default::default()
    };

    let red_sphere = Sphere {
        material: Material {
            color: Color::new(0.9, 0.08, 0.12),
            diffuse: 0.7,
            specular: 0.35,
            shininess: 100.0,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.65, 0.65, 0.65)),
            Transform::translation(Translation3::new(-2.0, 0.65, 1.2)),
        ]),
        ..Default::default()
    };

    let blue_sphere = Sphere {
        material: Material {
            color: Color::new(0.05, 0.25, 0.9),
            diffuse: 0.7,
            specular: 0.45,
            shininess: 150.0,
            reflective: 0.2,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.8, 0.8, 0.8)),
            Transform::translation(Translation3::new(2.1, 0.8, 0.6)),
        ]),
        ..Default::default()
    };

    let gold_sphere = Sphere {
        material: Material {
            color: Color::new(1.0, 0.62, 0.08),
            diffuse: 0.65,
            specular: 0.5,
            shininess: 200.0,
            reflective: 0.35,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.4, 0.4, 0.4)),
            Transform::translation(Translation3::new(0.0, 0.4, -2.2)),
        ]),
        ..Default::default()
    };

    let world = World::new(
        PointLight::new(Point3::new(-6.0, 8.0, -8.0), Color::new(1.0, 1.0, 1.0)),
        vec![
            Box::new(floor),
            Box::new(glass_sphere),
            Box::new(air_bubble),
            Box::new(red_sphere),
            Box::new(blue_sphere),
            Box::new(gold_sphere),
        ],
    );

    let camera = Camera::new(
        1000,
        600,
        std::f64::consts::PI / 3.0,
        World::view_transform(
            Point3::new(0.0, 3.5, -9.0),
            Point3::new(0.0, 1.2, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("refraction.ppm"))?;

    Ok(())
}
