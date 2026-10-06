use crate::{
    canvas::Color,
    light::{Lighting, PointLight},
    material::Material,
    ray::Ray,
    shape::{Intersection, Shape},
    transforms::{Transform, TransformError},
};

use nalgebra::{Point3, Vector3};

fn check_axis(origin: f64, direction: f64) -> (f64, f64) {
    const EPSILON: f64 = 1e-6;
    let tmin_numerator = -1.0 - origin;
    let tmax_numerator = 1.0 - origin;

    let (tmin, tmax) = if direction.abs() >= EPSILON {
        (tmin_numerator / direction, tmax_numerator / direction)
    } else {
        (
            tmin_numerator * f64::INFINITY,
            tmax_numerator * f64::INFINITY,
        )
    };

    if tmin > tmax {
        (tmax, tmin)
    } else {
        (tmin, tmax)
    }
}

#[derive(Debug)]
pub struct Cube {
    pub material: Material,
    pub transform: Transform,
}

impl Default for Cube {
    fn default() -> Self {
        Self {
            material: Material::default(),
            transform: Transform::identity(),
        }
    }
}

impl Cube {
    pub fn new(material: Material, transform: Transform) -> Self {
        Self {
            material,
            transform,
        }
    }
}

impl Lighting for Cube {
    fn lighting(
        &self,
        light: PointLight,
        point: Point3<f64>,
        eyev: Vector3<f64>,
        normalv: Vector3<f64>,
        in_shadow: bool,
    ) -> Result<Color, TransformError> {
        let effective_color = self.material.color * light.intensity;
        let lightv = (light.position - point).normalize();
        let ambient = effective_color * self.material.ambient;

        let light_dot_normal = lightv.dot(&normalv);
        let diffuse;
        let specular;

        if light_dot_normal < 0.0 || in_shadow {
            diffuse = Color::new(0.0, 0.0, 0.0);
            specular = Color::new(0.0, 0.0, 0.0);
        } else {
            diffuse = effective_color * self.material.diffuse * light_dot_normal;

            let reflectv = -lightv + 2.0 * light_dot_normal * normalv;
            let reflect_dot_eye = reflectv.dot(&eyev);

            if reflect_dot_eye <= 0.0 {
                specular = Color::new(0.0, 0.0, 0.0);
            } else {
                let factor = reflect_dot_eye.powf(self.material.shininess);
                specular = light.intensity * self.material.specular * factor;
            }
        }

        Ok(ambient + diffuse + specular)
    }
}

impl Shape for Cube {
    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn material(&self) -> &Material {
        &self.material
    }

    fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    fn intersect(&self, ray: &Ray) -> Result<Vec<Intersection<'_>>, TransformError> {
        let ray = Ray::new(
            self.transform().apply_inverse(ray.origin)?,
            self.transform().apply_inverse(ray.direction)?,
        );

        let (xtmin, xtmax) = check_axis(ray.origin.x, ray.direction.x);
        let (ytmin, ytmax) = check_axis(ray.origin.y, ray.direction.y);
        let (ztmin, ztmax) = check_axis(ray.origin.z, ray.direction.z);

        let tmin = xtmin.max(ytmin).max(ztmin);
        let tmax = xtmax.min(ytmax).min(ztmax);

        if tmin > tmax {
            Ok(vec![])
        } else {
            Ok(vec![
                Intersection::new(tmin, self),
                Intersection::new(tmax, self),
            ])
        }
    }

    fn normal_at(&self, world_point: Point3<f64>) -> Result<Vector3<f64>, TransformError> {
        let object_point = self.transform.apply_inverse(world_point)?;
        let maxc = object_point
            .x
            .abs()
            .max(object_point.y.abs())
            .max(object_point.z.abs());

        let object_normal = if maxc == object_point.x.abs() {
            Vector3::new(object_point.x, 0.0, 0.0)
        } else if maxc == object_point.y.abs() {
            Vector3::new(0.0, object_point.y, 0.0)
        } else {
            Vector3::new(0.0, 0.0, object_point.z)
        };

        let world_normal = self.transform.apply_transpose_inverse(object_normal)?;
        Ok(world_normal.normalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_a_ray_intersects_a_cube() {
        let c = Cube::default();
        let rays = [
            Ray::new(Point3::new(5.0, 0.5, 0.0), Vector3::new(-1.0, 0.0, 0.0)),
            Ray::new(Point3::new(-5.0, 0.5, 0.0), Vector3::new(1.0, 0.0, 0.0)),
            Ray::new(Point3::new(0.5, 5.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
            Ray::new(Point3::new(0.5, -5.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            Ray::new(Point3::new(0.5, 0.0, 5.0), Vector3::new(0.0, 0.0, -1.0)),
            Ray::new(Point3::new(0.5, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
            Ray::new(Point3::new(0.0, 0.5, 0.0), Vector3::new(0.0, 0.0, 1.0)),
        ];
        let expected_ts = [
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (4.0, 6.0),
            (-1.0, 1.0),
        ];

        for (ray, (expected_t1, expected_t2)) in rays.iter().zip(expected_ts.iter()) {
            let xs = c.intersect(ray).unwrap();
            assert_eq!(xs.len(), 2);
            assert_abs_diff_eq!(xs[0].t, *expected_t1);
            assert_abs_diff_eq!(xs[1].t, *expected_t2);
        }
    }

    #[test]
    fn test_the_normal_on_the_surface_of_a_cube() {
        let c = Cube::default();
        let points = [
            Point3::new(1.0, 0.5, -0.8),
            Point3::new(-1.0, -0.2, 0.9),
            Point3::new(-0.4, 1.0, -0.1),
            Point3::new(0.3, -1.0, -0.7),
            Point3::new(-0.6, 0.3, 1.0),
            Point3::new(0.4, 0.4, -1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, -1.0, -1.0),
        ];
        let expected_normals = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];

        for (point, expected_normal) in points.iter().zip(expected_normals.iter()) {
            let normal = c.normal_at(*point).unwrap();
            assert_abs_diff_eq!(normal.x, expected_normal.x);
            assert_abs_diff_eq!(normal.y, expected_normal.y);
            assert_abs_diff_eq!(normal.z, expected_normal.z);
        }
    }
}
