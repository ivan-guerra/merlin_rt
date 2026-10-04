use merlin_rt::{
    camera::Camera, canvas::Color, light::PointLight, material::Material, pattern::CheckerPattern,
    plane::Plane, scene::World, sphere::Sphere, transforms::Transform,
};

use anyhow::Result;
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    let floor = Plane {
        material: Material {
            ambient: 0.2,
            diffuse: 0.8,
            specular: 0.0,
            pattern: Some(Box::new(CheckerPattern {
                a: Color::new(1.0, 1.0, 1.0),
                b: Color::new(0.1, 0.1, 0.1),
                ..Default::default()
            })),
            ..Default::default()
        },
        transform: Transform::identity(),
    };

    let glass_sphere = Sphere {
        material: Material {
            color: Color::new(0.95, 0.95, 1.0),
            ambient: 0.0,
            diffuse: 0.1,
            specular: 0.9,
            shininess: 300.0,
            reflective: 0.2,
            transparency: 1.0,
            refractive_index: 1.5,
            ..Default::default()
        },
        transform: Transform::translation(Translation3::new(0.0, 1.0, 0.0)),
        ..Default::default()
    };

    let air_bubble = Sphere {
        material: Material {
            ambient: 0.0,
            diffuse: 0.0,
            specular: 0.0,
            transparency: 1.0,
            refractive_index: 1.0,
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::scale(Scale3::new(0.35, 0.35, 0.35)),
            Transform::translation(Translation3::new(0.0, 1.0, 0.0)),
        ]),
        ..Default::default()
    };

    let world = World::new(
        PointLight::new(Point3::new(-10.0, 10.0, -10.0), Color::new(1.0, 1.0, 1.0)),
        vec![
            Box::new(floor),
            Box::new(glass_sphere),
            Box::new(air_bubble),
        ],
    );

    let camera = Camera::new(
        1000,
        500,
        std::f64::consts::PI / 3.0,
        World::view_transform(
            Point3::new(0.0, 6.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("refraction.ppm"))?;

    Ok(())
}
