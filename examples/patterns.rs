use anyhow::Result;
use merlin_rt::{
    geometry::{
        shapes::{Plane, Sphere},
        transforms::{Axis, Transform},
    },
    rendering::{camera::Camera, canvas::Color},
    scene::{
        light::PointLight,
        material::Material,
        pattern::{CheckerPattern, GradientPattern, StripePattern},
        world::World,
    },
};
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::path::Path;

fn main() -> Result<()> {
    let floor = Plane {
        material: Material {
            ambient: 0.2,
            diffuse: 0.75,
            specular: 0.15,
            pattern: Some(Box::new(CheckerPattern {
                a: Color::new(0.9, 0.9, 0.9),
                b: Color::new(0.15, 0.15, 0.18),
                transform: Transform::scale(Scale3::new(0.5, 0.5, 0.5)),
            })),
            ..Default::default()
        },
        ..Default::default()
    };

    let striped_sphere = Sphere {
        material: Material {
            ambient: 0.1,
            diffuse: 0.7,
            specular: 0.4,
            shininess: 100.0,
            pattern: Some(Box::new(StripePattern {
                a: Color::new(0.95, 0.25, 0.1),
                b: Color::new(1.0, 0.9, 0.2),
                transform: Transform::scale(Scale3::new(0.25, 0.25, 0.25)),
            })),
            ..Default::default()
        },
        transform: Transform::translation(Translation3::new(-2.1, 1.0, 0.5)),
        ..Default::default()
    };

    let gradient_wall = Plane {
        material: Material {
            ambient: 0.25,
            diffuse: 0.65,
            specular: 0.05,
            pattern: Some(Box::new(GradientPattern {
                a: Color::new(0.08, 0.55, 0.95),
                b: Color::new(0.2, 0.03, 0.45),
                // Enlarge pattern space to show one broad gradient.
                transform: Transform::scale(Scale3::new(5.0, 1.0, 1.0)),
            })),
            ..Default::default()
        },
        transform: Transform::sequence([
            Transform::rotation(Axis::X, std::f64::consts::FRAC_PI_2),
            Transform::translation(Translation3::new(0.0, 0.0, 4.0)),
        ]),
    };

    let checker_sphere = Sphere {
        material: Material {
            ambient: 0.1,
            diffuse: 0.7,
            specular: 0.4,
            shininess: 100.0,
            pattern: Some(Box::new(CheckerPattern {
                a: Color::new(0.1, 0.9, 0.35),
                b: Color::new(0.03, 0.2, 0.1),
                transform: Transform::scale(Scale3::new(0.35, 0.35, 0.35)),
            })),
            ..Default::default()
        },
        transform: Transform::translation(Translation3::new(2.1, 1.0, 0.5)),
        ..Default::default()
    };

    let world = World::new(
        PointLight::new(Point3::new(-6.0, 7.0, -8.0), Color::new(1.0, 1.0, 1.0)),
        vec![
            Box::new(floor),
            Box::new(striped_sphere),
            Box::new(gradient_wall),
            Box::new(checker_sphere),
        ],
    );

    let camera = Camera::new(
        1000,
        600,
        std::f64::consts::PI / 3.0,
        World::view_transform(
            Point3::new(0.0, 3.0, -8.0),
            Point3::new(0.0, 1.0, 0.5),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("patterns.ppm"))?;

    Ok(())
}
