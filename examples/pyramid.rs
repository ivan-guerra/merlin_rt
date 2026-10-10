use anyhow::Result;
use merlin_rt::{
    geometry::shapes::{Group, Plane, Triangle},
    rendering::{camera::Camera, canvas::Color},
    scene::{light::PointLight, material::Material, world::World},
};
use nalgebra::{Point3, Vector3};
use std::{f64::consts::PI, path::Path, rc::Rc};

fn pyramid() -> Result<Rc<Group>> {
    let apex = Point3::new(0.0, 2.0, 0.0);
    let a = Point3::new(-1.0, 0.0, -1.0);
    let b = Point3::new(1.0, 0.0, -1.0);
    let c = Point3::new(1.0, 0.0, 1.0);
    let d = Point3::new(-1.0, 0.0, 1.0);
    let mut group = Group::builder();

    // Four sides and two base triangles, wound for outward-facing normals.
    for [p1, p2, p3] in [
        [apex, a, b],
        [apex, b, c],
        [apex, c, d],
        [apex, d, a],
        [a, c, b],
        [a, d, c],
    ] {
        // Materials belong to the triangles, not the enclosing group.
        group = group.child(Rc::new(
            Triangle::builder()
                .vertices(p1, p2, p3)
                .material(Material {
                    color: Color::new(0.85, 0.6, 0.25),
                    diffuse: 0.7,
                    specular: 0.2,
                    shininess: 80.0,
                    ..Default::default()
                })
                .build(),
        ));
    }

    Ok(group.build()?)
}

fn main() -> Result<()> {
    let pyramid = pyramid()?;
    let floor = Plane::builder()
        .material(Material {
            color: Color::new(0.9, 0.9, 0.9),
            ambient: 0.15,
            diffuse: 0.7,
            specular: 0.0,
            ..Default::default()
        })
        .build();

    // Keep the root group alive so the triangles can resolve their parent links.
    let world = World::from_shared(
        PointLight::new(Point3::new(-5.0, 6.0, -8.0), Color::new(1.0, 1.0, 1.0)),
        vec![Rc::new(floor), pyramid],
    );
    let camera = Camera::new(
        400,
        300,
        PI / 3.0,
        World::view_transform(
            Point3::new(4.0, 3.0, -6.0),
            Point3::new(0.0, 0.9, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    );

    camera
        .render(&world)?
        .write_to_ppm(Path::new("pyramid.ppm"))?;
    Ok(())
}
