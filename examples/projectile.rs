use merlin_rt::primitives::{Point, Vector};

use anyhow::Result;
use std::io::Write;

#[derive(Debug)]
struct Projectile {
    position: Point,
    velocity: Vector,
}

#[derive(Debug)]
struct Environment {
    gravity: Vector,
    wind: Vector,
}

fn tick(env: &Environment, proj: &Projectile) -> Projectile {
    let position = proj.position + proj.velocity;
    let velocity = proj.velocity + env.gravity + env.wind;
    Projectile { position, velocity }
}

fn main() -> Result<()> {
    let mut input = String::new();
    print!("Enter the scale of the velocity vector: ");
    std::io::stdout().flush().expect("Failed to flush stdout");
    std::io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    let scale: f64 = input
        .trim()
        .parse()
        .expect("Please enter a valid floating point number");
    let p = Projectile {
        // Projectile starts one unit above the origin.
        position: Point::new(0.0, 1.0, 0.0),
        // Velocity is normalized to one unit per tick.
        velocity: Vector::new(1.0, 1.0, 0.0) * scale,
    };
    println!("Initial Position: {}", p.position);
    println!("Initial Velocity (post normalization): {}", p.velocity);

    let e = Environment {
        gravity: Vector::new(0.0, -0.1, 0.0),
        wind: Vector::new(-0.01, 0.0, 0.0),
    };
    println!("Gravity: {}", e.gravity);
    println!("Wind: {}", e.wind);

    println!("\nStarting simulation...");

    let mut projectile = p;
    let mut i = 0;
    while projectile.position.y > 0.0 {
        println!("Tick #{}: {}", i, projectile.position);
        projectile = tick(&e, &projectile);
        i += 1;
    }
    println!("It took {} ticks for the projectile to hit the ground.", i);

    Ok(())
}
