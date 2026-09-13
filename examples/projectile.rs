use merlin_rt::canvas::{Canvas, Color};
use merlin_rt::primitives::{Point3, Vec3};

use anyhow::Result;
use std::path::Path;

#[derive(Debug)]
struct Projectile {
    position: Point3,
    velocity: Vec3,
}

#[derive(Debug)]
struct Environment {
    gravity: Vec3,
    wind: Vec3,
}

fn tick(env: &Environment, proj: &Projectile, canvas: &mut Canvas) -> Result<Projectile> {
    let projectile_color = Color::new(0.0, 1.0, 0.0);
    let position = proj.position + proj.velocity;
    let velocity = proj.velocity + env.gravity + env.wind;
    let x = position.x as usize;
    let y = canvas.height.saturating_sub(position.y as usize);

    if x < canvas.width && y < canvas.height {
        canvas.write_pixel(x, y, projectile_color)?;
    }

    Ok(Projectile { position, velocity })
}

fn main() -> Result<()> {
    let mut canvas = Canvas::new(900, 550);
    let p = Projectile {
        // Projectile starts one unit above the origin.
        position: Point3::new(0.0, 1.0, 0.0),
        // Velocity is normalized to one unit per tick.
        velocity: Vec3::new(1.0, 1.8, 0.0).normalize()? * 11.25,
    };
    let e = Environment {
        gravity: Vec3::new(0.0, -0.1, 0.0),
        wind: Vec3::new(-0.01, 0.0, 0.0),
    };

    println!("Initial Position: {}", p.position);
    println!("Initial Velocity (post normalization): {}", p.velocity);
    println!("Gravity: {}", e.gravity);
    println!("Wind: {}", e.wind);
    println!("\nStarting simulation...");

    let mut projectile = p;
    let mut i = 0;
    while projectile.position.y > 0.0 {
        println!("Tick #{}: {}", i, projectile.position);
        projectile = tick(&e, &projectile, &mut canvas)?;
        i += 1;
    }
    println!("It took {} ticks for the projectile to hit the ground.", i);

    canvas.write_to_ppm(Path::new("projectile.ppm"))?;

    Ok(())
}
