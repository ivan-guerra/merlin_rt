use anyhow::Result;
use merlin_rt::{
    camera::Camera,
    canvas::Color,
    light::PointLight,
    material::Material,
    pattern::CheckerPattern,
    scene::World,
    shapes::{Cube, Plane, Shape},
    transforms::Transform,
};
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::path::Path;

fn wood() -> Material {
    Material {
        color: Color::new(0.45, 0.22, 0.08),
        ambient: 0.15,
        diffuse: 0.7,
        specular: 0.25,
        shininess: 80.0,
        ..Default::default()
    }
}

fn cube(scale: Vector3<f64>, translation: Vector3<f64>) -> Cube {
    Cube::new(
        wood(),
        Transform::sequence([
            Transform::scale(Scale3::new(scale.x, scale.y, scale.z)),
            Transform::translation(Translation3::new(
                translation.x,
                translation.y,
                translation.z,
            )),
        ]),
    )
}

fn main() -> Result<()> {
    let floor = Plane {
        material: Material {
            ambient: 0.2,
            diffuse: 0.8,
            specular: 0.05,
            pattern: Some(Box::new(CheckerPattern {
                a: Color::new(0.12, 0.13, 0.16),
                b: Color::new(0.42, 0.44, 0.48),
                transform: Transform::scale(Scale3::new(0.75, 0.75, 0.75)),
            })),
            ..Default::default()
        },
        transform: Transform::identity(),
    };

    let tabletop = cube(Vector3::new(3.0, 0.2, 2.0), Vector3::new(0.0, 2.1, 0.0));

    let leg_positions = [
        Vector3::new(-2.55, 1.0, -1.55),
        Vector3::new(2.55, 1.0, -1.55),
        Vector3::new(-2.55, 1.0, 1.55),
        Vector3::new(2.55, 1.0, 1.55),
    ];
    let legs = leg_positions
        .into_iter()
        .map(|position| Box::new(cube(Vector3::new(0.22, 1.0, 0.22), position)) as Box<dyn Shape>);

    let centerpiece = cube(Vector3::new(0.35, 0.35, 0.35), Vector3::new(0.0, 2.65, 0.0));

    let mut objects: Vec<Box<dyn Shape>> =
        vec![Box::new(floor), Box::new(tabletop), Box::new(centerpiece)];
    objects.extend(legs);

    let world = World::new(
        PointLight::new(Point3::new(-6.0, 8.0, -8.0), Color::new(1.0, 1.0, 1.0)),
        objects,
    );

    let camera = Camera::new(
        800,
        600,
        std::f64::consts::PI / 3.0,
        World::view_transform(
            Point3::new(7.0, 5.0, -9.0),
            Point3::new(0.0, 1.3, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    let canvas = camera.render(&world)?;
    canvas.write_to_ppm(Path::new("cube.ppm"))?;

    Ok(())
}
