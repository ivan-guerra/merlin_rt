use anyhow::Result;
use merlin_rt::{
    geometry::{
        shapes::{
            Csg, CsgOperation, Cube, Cylinder, DoubleNappedCone, Plane, Shape, ShapeRef, Sphere,
        },
        transforms::{Axis, Transform},
    },
    rendering::{camera::Camera, canvas::Color},
    scene::{light::PointLight, material::Material, pattern::CheckerPattern, world::World},
};
use nalgebra::{Point3, Scale3, Translation3, Vector3};
use std::{f64::consts::TAU, path::Path, rc::Rc};

fn surface(color: Color) -> Material {
    Material {
        color,
        diffuse: 0.75,
        specular: 0.2,
        shininess: 80.0,
        ..Default::default()
    }
}

fn gold() -> Material {
    Material {
        color: Color::new(0.85, 0.55, 0.12),
        diffuse: 0.6,
        specular: 0.7,
        reflective: 0.15,
        ..Default::default()
    }
}

fn magic() -> Material {
    // Ambient color suggests a glow; materials do not emit light or bloom.
    Material {
        color: Color::new(0.2, 0.85, 1.0),
        ambient: 0.8,
        diffuse: 0.2,
        specular: 0.8,
        ..Default::default()
    }
}

fn glass_orb() -> Material {
    Material {
        color: Color::new(0.5, 0.85, 1.0),
        ambient: 0.03,
        diffuse: 0.1,
        specular: 1.0,
        shininess: 300.0,
        reflective: 0.9,
        transparency: 0.9,
        refractive_index: 1.5,
        ..Default::default()
    }
}

fn placed(scale: [f64; 3], tilt: f64, position: [f64; 3]) -> Transform {
    let [sx, sy, sz] = scale;
    let [x, y, z] = position;
    Transform::sequence([
        Transform::scale(Scale3::new(sx, sy, sz)),
        Transform::rotation(Axis::Z, tilt),
        Transform::translation(Translation3::new(x, y, z)),
    ])
}

fn sphere(scale: [f64; 3], tilt: f64, position: [f64; 3], material: Material) -> Box<dyn Shape> {
    Box::new(
        Sphere::builder()
            .transform(placed(scale, tilt, position))
            .material(material)
            .build(),
    )
}

fn cylinder(scale: [f64; 3], position: [f64; 3], material: Material) -> Box<dyn Shape> {
    Box::new(
        Cylinder::builder()
            .minimum(0.0)
            .maximum(1.0)
            .closed(true)
            .transform(placed(scale, 0.0, position))
            .material(material)
            .build(),
    )
}

// Cubes use half-extents; yaw keeps furniture aligned with the tower wall.
fn cuboid(scale: [f64; 3], yaw: f64, position: [f64; 3], material: Material) -> Box<dyn Shape> {
    let [sx, sy, sz] = scale;
    let [x, y, z] = position;
    Box::new(
        Cube::builder()
            .transform(Transform::sequence([
                Transform::scale(Scale3::new(sx, sy, sz)),
                Transform::rotation(Axis::Y, yaw),
                Transform::translation(Translation3::new(x, y, z)),
            ]))
            .material(material)
            .build(),
    )
}

fn tower(objects: &mut Vec<Box<dyn Shape>>) {
    // Enclose the camera and key light as well as the scene. The outer wall and
    // ceiling catch rays through the open sides, including glass reflections.
    let height = 10.4;
    objects.push(cylinder(
        [16.0, height + 0.1, 16.0],
        [0.0, -0.1, -0.5],
        Material {
            color: Color::new(0.24, 0.25, 0.31),
            ambient: 0.25,
            diffuse: 0.65,
            specular: 0.05,
            ..Default::default()
        },
    ));

    // Detailed rear masonry extends to the ceiling; the foreground stays open.
    let segments = 11;
    let arc = std::f64::consts::PI / f64::from(segments);
    let half_width = 6.0 * (arc / 2.0).tan();
    for segment in 0..segments {
        let angle = -std::f64::consts::FRAC_PI_2 + (f64::from(segment) + 0.5) * arc;
        objects.push(cuboid(
            [half_width + 0.06, height / 2.0, 0.18],
            angle,
            [6.2 * angle.sin(), height / 2.0, -0.5 + 6.2 * angle.cos()],
            surface(Color::new(0.09, 0.1, 0.14)),
        ));
        for row in 0..10 {
            let shade = 0.025 * f64::from((segment + row * 2) % 3);
            objects.push(cuboid(
                [half_width - 0.025, 0.48, 0.16],
                angle,
                [
                    6.0 * angle.sin(),
                    0.51 + f64::from(row) * 1.03,
                    -0.5 + 6.0 * angle.cos(),
                ],
                Material {
                    color: Color::new(0.24 + shade, 0.25 + shade, 0.31 + shade),
                    ambient: 0.25,
                    diffuse: 0.65,
                    specular: 0.05,
                    ..Default::default()
                },
            ));
        }
    }
}

fn shelf_book(objects: &mut Vec<Box<dyn Shape>>, x: f64, bottom: f64, z: f64, index: usize) {
    let colors = [
        Color::new(0.42, 0.09, 0.1),
        Color::new(0.08, 0.3, 0.22),
        Color::new(0.12, 0.18, 0.42),
        Color::new(0.4, 0.24, 0.08),
        Color::new(0.29, 0.1, 0.36),
    ];
    let height = 0.7 + (index % 3) as f64 * 0.12;
    objects.push(cuboid(
        [0.12, height / 2.0, 0.24],
        0.0,
        [x, bottom + height / 2.0, z],
        Material {
            ambient: 0.22,
            ..surface(colors[index % colors.len()])
        },
    ));
    // Gold tooling faces the camera on the spine, not the hidden page edges.
    for y in [bottom + 0.12, bottom + height - 0.12] {
        objects.push(cuboid([0.1, 0.025, 0.012], 0.0, [x, y, z - 0.245], gold()));
    }
}

fn potion(objects: &mut Vec<Box<dyn Shape>>, position: [f64; 3], color: Color) {
    let [x, bottom, z] = position;
    // Opaque, jewel-colored bottles avoid the renderer's opaque glass shadows.
    objects.push(sphere(
        [0.18, 0.22, 0.18],
        0.0,
        [x, bottom + 0.22, z],
        Material {
            color,
            ambient: 0.4,
            diffuse: 0.4,
            specular: 0.9,
            shininess: 150.0,
            ..Default::default()
        },
    ));
    objects.push(cylinder(
        [0.075, 0.17, 0.075],
        [x, bottom + 0.37, z],
        surface(color),
    ));
    objects.push(cylinder([0.09, 0.07, 0.09], [x, bottom + 0.54, z], gold()));
}

fn bookshelf(objects: &mut Vec<Box<dyn Shape>>, x: f64, z: f64, palette_offset: usize) {
    let wood = Color::new(0.22, 0.105, 0.045);
    objects.push(cuboid(
        [1.12, 2.15, 0.055],
        0.0,
        [x, 2.4, z + 0.35],
        surface(Color::new(0.12, 0.055, 0.025)),
    ));
    for side in [-1.12, 1.12] {
        objects.push(cuboid(
            [0.075, 2.15, 0.4],
            0.0,
            [x + side, 2.4, z],
            surface(wood),
        ));
    }
    for (level, y) in [0.35, 1.65, 2.95, 4.35].into_iter().enumerate() {
        objects.push(cuboid([1.2, 0.075, 0.4], 0.0, [x, y, z], surface(wood)));
        if level < 2 {
            for book in 0..5 {
                shelf_book(
                    objects,
                    x - 0.8 + book as f64 * 0.34,
                    y + 0.075,
                    z - 0.08,
                    book + level + palette_offset,
                );
            }
        }
    }
    // The upper shelf mixes grimoires with alchemical bottles and a crystal relic.
    for book in 0..2 {
        shelf_book(
            objects,
            x - 0.82 + book as f64 * 0.32,
            3.025,
            z - 0.08,
            book + palette_offset + 3,
        );
    }
    potion(
        objects,
        [x - 0.08, 3.025, z - 0.08],
        Color::new(0.18, 0.7, 0.38),
    );
    potion(
        objects,
        [x + 0.4, 3.025, z - 0.08],
        Color::new(0.65, 0.15, 0.55),
    );
    objects.push(cylinder(
        [0.18, 0.08, 0.18],
        [x + 0.85, 3.025, z - 0.08],
        gold(),
    ));
    objects.push(Box::new(
        DoubleNappedCone::builder()
            .minimum(-1.0)
            .maximum(0.0)
            .closed(true)
            .transform(placed([0.16, 0.55, 0.16], 0.0, [x + 0.85, 3.65, z - 0.08]))
            .material(magic())
            .build(),
    ));
    // A small brass planetary model crowns each bookcase.
    objects.push(cylinder([0.3, 0.08, 0.3], [x, 4.425, z], gold()));
    objects.push(cylinder([0.045, 0.38, 0.045], [x, 4.505, z], gold()));
    objects.push(sphere([0.18; 3], 0.0, [x, 4.98, z], glass_orb()));
    for step in 0..6 {
        let angle = f64::from(step) * TAU / 6.0;
        objects.push(sphere(
            [0.07; 3],
            0.0,
            [
                x + 0.38 * angle.cos(),
                4.98 + 0.2 * angle.sin(),
                z + 0.24 * angle.sin(),
            ],
            gold(),
        ));
    }
}

fn buttoned_robe(material: Material) -> Result<ShapeRef> {
    // Robe: the lower half of a cone, truncated at the shoulders.
    let mut robe: ShapeRef = Rc::new(
        DoubleNappedCone::builder()
            .minimum(-1.0)
            .maximum(-0.25)
            .closed(true)
            .transform(placed([0.85, 2.5, 0.65], 0.0, [-0.6, 2.55, 0.2]))
            .material(material)
            .build(),
    );
    for y in [0.7, 1.05, 1.4] {
        // Recess the button center slightly into the cone's front surface.
        let front_z = 0.2 - 0.65 * (2.55 - y) / 2.5;
        let button = Rc::from(sphere(
            [0.055, 0.055, 0.035],
            0.0,
            [-0.6, y, front_z + 0.01],
            gold(),
        ));
        // Union hides internal boundaries and preserves exposed gold surfaces.
        robe = Csg::new(CsgOperation::Union, robe, button)?;
    }
    Ok(robe)
}

fn scene() -> Result<World> {
    let robe = Color::new(0.16, 0.07, 0.38);
    let skin = Color::new(0.85, 0.59, 0.4);
    let beard = Color::new(0.85, 0.88, 0.95);
    let stone = Color::new(0.16, 0.19, 0.27);
    let mut objects: Vec<Box<dyn Shape>> = vec![
        Box::new(
            Plane::builder()
                .material(Material {
                    pattern: Some(Box::new(CheckerPattern {
                        a: Color::new(0.09, 0.11, 0.17),
                        b: Color::new(0.18, 0.2, 0.28),
                        ..Default::default()
                    })),
                    specular: 0.1,
                    reflective: 0.12,
                    ..Default::default()
                })
                .build(),
        ),
        sphere([0.38, 0.46, 0.35], 0.0, [-0.6, 2.6, 0.0], surface(skin)),
        // A beard widens toward the face, with its point below the chin.
        Box::new(
            DoubleNappedCone::builder()
                .minimum(0.0)
                .maximum(1.0)
                .closed(true)
                .transform(placed([0.32, 0.9, 0.2], 0.0, [-0.6, 1.65, -0.35]))
                .material(surface(beard))
                .build(),
        ),
        sphere([0.09, 0.12, 0.12], 0.0, [-0.6, 2.64, -0.36], surface(skin)),
        cylinder([0.72, 0.09, 0.65], [-0.6, 2.98, 0.0], surface(robe)),
        // The upper point leans slightly; only one nappe forms the hat.
        Box::new(
            DoubleNappedCone::builder()
                .minimum(-1.0)
                .maximum(0.0)
                .closed(true)
                .transform(placed([0.58, 1.35, 0.55], -0.12, [-0.44, 4.4, 0.0]))
                .material(surface(robe))
                .build(),
        ),
        cylinder([0.58, 0.1, 0.55], [-0.6, 3.06, 0.0], gold()),
        sphere([0.27, 0.6, 0.28], -0.65, [-1.15, 1.98, 0.0], surface(robe)),
        sphere([0.27, 0.55, 0.28], 0.8, [-0.06, 2.05, 0.0], surface(robe)),
        sphere([0.15, 0.17, 0.15], 0.0, [-1.55, 2.25, -0.16], surface(skin)),
        sphere([0.15, 0.17, 0.15], 0.0, [0.33, 2.34, -0.12], surface(skin)),
        // Wooden staff with a reflective glass orb resting on its gold collar.
        cylinder(
            [0.065, 3.25, 0.065],
            [-1.65, 0.05, -0.22],
            surface(Color::new(0.24, 0.1, 0.04)),
        ),
        cylinder([0.13, 0.1, 0.13], [-1.65, 3.25, -0.22], gold()),
        sphere([0.3; 3], 0.0, [-1.65, 3.65, -0.22], glass_orb()),
        // Crystal orb on a stone pedestal.
        cylinder([0.62, 0.12, 0.62], [1.35, 0.02, -0.1], surface(stone)),
        cylinder([0.24, 0.7, 0.24], [1.35, 0.14, -0.1], surface(stone)),
        cylinder([0.5, 0.12, 0.5], [1.35, 0.84, -0.1], gold()),
        sphere(
            [0.43, 0.43, 0.43],
            0.0,
            [1.35, 1.38, -0.1],
            Material {
                color: Color::new(0.25, 0.65, 0.85),
                ambient: 0.2,
                diffuse: 0.15,
                specular: 0.9,
                reflective: 0.25,
                transparency: 0.7,
                refractive_index: 1.5,
                ..Default::default()
            },
        ),
    ];

    for x in [-0.75, -0.45] {
        objects.push(sphere(
            [0.045, 0.055, 0.035],
            0.0,
            [x, 2.77, -0.32],
            surface(Color::new(0.025, 0.02, 0.04)),
        ));
    }
    for y in [2.95, 3.18, 3.25] {
        objects.push(cylinder([0.1, 0.07, 0.1], [-1.65, y, -0.22], gold()));
    }
    // Three spellbooks: each has a page block between two colored covers.
    for (level, color) in [
        Color::new(0.35, 0.07, 0.08),
        Color::new(0.05, 0.24, 0.2),
        Color::new(0.16, 0.09, 0.32),
    ]
    .into_iter()
    .enumerate()
    {
        let y = 0.13 + level as f64 * 0.23;
        for (offset, height, color) in [
            (-0.095, 0.025, color),
            (0.0, 0.07, Color::new(0.75, 0.68, 0.5)),
            (0.095, 0.025, color),
        ] {
            objects.push(Box::new(
                Cube::builder()
                    .transform(Transform::sequence([
                        Transform::scale(Scale3::new(0.38, height, 0.27)),
                        Transform::rotation(Axis::Y, level as f64 * 0.2 - 0.15),
                        Transform::translation(Translation3::new(2.0, y + offset, -1.15)),
                    ]))
                    .material(surface(color))
                    .build(),
            ));
        }
    }

    tower(&mut objects);
    bookshelf(&mut objects, -3.6, 2.8, 0);
    bookshelf(&mut objects, 1.35, 3.8, 2);

    let mut objects: Vec<ShapeRef> = objects.into_iter().map(Rc::from).collect();
    objects.insert(1, buttoned_robe(surface(robe))?);
    Ok(World::from_shared(
        PointLight::new(Point3::new(-3.5, 7.0, -6.0), Color::new(1.0, 0.88, 0.72)),
        objects,
    ))
}

fn camera() -> Camera {
    // Face the rear masonry nearly straight on, keeping the open tower sides
    // and outer enclosure outside the frame while retaining both bookcases.
    Camera::new(
        1920,
        1080,
        std::f64::consts::PI * 65.0 / 180.0,
        World::view_transform(
            Point3::new(0.0, 3.4, -8.0),
            Point3::new(0.0, 2.6, 1.0),
            Vector3::y(),
        ),
    )
}

fn main() -> Result<()> {
    camera()
        .render(&scene()?)?
        .write_to_ppm(Path::new("wizard.ppm"))?;
    Ok(())
}
