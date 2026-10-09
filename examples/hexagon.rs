use merlin_rt::{
    geometry::{
        shapes::{Cylinder, Group, Plane, ShapeRef, Sphere},
        transforms::{Axis, Transform},
    },
    rendering::{camera::Camera, canvas::Color},
    scene::{light::PointLight, material::Material, world::World},
};

use anyhow::Result;
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::{f64::consts::PI, path::Path, rc::Rc};

const CORNER_RADIUS: f64 = 0.12;
const EDGE_RADIUS: f64 = 0.07;

fn frame_material() -> Material {
    // Materials belong to the leaves; a group's material is not inherited.
    Material {
        color: Color::new(0.15, 0.55, 0.85),
        diffuse: 0.7,
        specular: 0.4,
        shininess: 100.0,
        ..Default::default()
    }
}

fn hexagon_corner() -> ShapeRef {
    Rc::new(
        Sphere::builder()
            .material(frame_material())
            .transform(Transform::sequence([
                Transform::scale(Scale3::new(CORNER_RADIUS, CORNER_RADIUS, CORNER_RADIUS)),
                Transform::translation(Translation3::new(0.0, 0.0, -1.0)),
            ]))
            .build(),
    )
}

fn hexagon_edge() -> ShapeRef {
    Rc::new(
        Cylinder::builder()
            .minimum(0.0)
            .maximum(1.0)
            .closed(true)
            .material(frame_material())
            .transform(Transform::sequence([
                // Applied in order: scale, rotate into the XZ plane, then position.
                // The endpoints are (0, 0, -1) and (-sqrt(3)/2, 0, -1/2).
                Transform::scale(Scale3::new(EDGE_RADIUS, 1.0, EDGE_RADIUS)),
                Transform::rotation(Axis::Z, PI / 2.0),
                Transform::rotation(Axis::Y, PI / 6.0),
                Transform::translation(Translation3::new(0.0, 0.0, -1.0)),
            ]))
            .build(),
    )
}

fn hexagon() -> Result<Rc<Group>> {
    let mut ring = Group::builder().transform(Transform::sequence([
        Transform::rotation(Axis::Y, PI / 12.0),
        Transform::translation(Translation3::new(0.0, CORNER_RADIUS, 0.0)),
    ]));

    for side in 0..6 {
        // Each side needs fresh shapes: children may have only one live parent.
        let side = Group::builder()
            .transform(Transform::rotation(Axis::Y, f64::from(side) * PI / 3.0))
            .child(hexagon_corner())
            .child(hexagon_edge())
            .build()?;
        ring = ring.child(side);
    }

    Ok(ring.build()?)
}

fn main() -> Result<()> {
    let hexagon = hexagon()?;
    let floor = Plane::builder()
        .material(Material {
            color: Color::new(0.9, 0.9, 0.9),
            ambient: 0.15,
            diffuse: 0.7,
            specular: 0.0,
            ..Default::default()
        })
        .build();

    // The world owns the root group, keeping all weak ancestor links resolvable.
    let world = World::from_shared(
        PointLight::new(Point3::new(-5.0, 6.0, -8.0), Color::new(1.0, 1.0, 1.0)),
        vec![Rc::new(floor), hexagon],
    );
    let camera = Camera::new(
        400,
        300,
        PI / 3.0,
        World::view_transform(
            Point3::new(0.0, 3.0, -2.0),
            Point3::new(0.0, CORNER_RADIUS, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    camera
        .render(&world)?
        .write_to_ppm(Path::new("hexagon.ppm"))?;
    Ok(())
}
