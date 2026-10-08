use crate::{
    geometry::{
        ray::Ray,
        transforms::{Transform, TransformError},
    },
    rendering::canvas::{Canvas, CanvasError},
    scene::world::{MAX_RECURSION_DEPTH, World},
};

use nalgebra::Point3;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error(transparent)]
    Transform(#[from] TransformError),

    #[error(transparent)]
    Canvas(#[from] CanvasError),
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub hsize: usize,
    pub vsize: usize,
    pub field_of_view: f64,
    pub transform: Transform,
    pixel_size: f64,
    half_width: f64,
    half_height: f64,
}

impl Camera {
    pub fn new(hsize: usize, vsize: usize, field_of_view: f64, transform: Transform) -> Self {
        let half_view = (field_of_view / 2.0).tan();
        let aspect = hsize as f64 / vsize as f64;
        let (half_width, half_height) = if aspect >= 1.0 {
            (half_view, half_view / aspect)
        } else {
            (half_view * aspect, half_view)
        };
        let pixel_size = (half_width * 2.0) / hsize as f64;

        Camera {
            hsize,
            vsize,
            field_of_view,
            transform,
            pixel_size,
            half_width,
            half_height,
        }
    }

    pub fn ray_for_pixel(&self, px: usize, py: usize) -> Result<Ray, TransformError> {
        let xoffset = (px as f64 + 0.5) * self.pixel_size;
        let yoffset = (py as f64 + 0.5) * self.pixel_size;

        let world_x = self.half_width - xoffset;
        let world_y = self.half_height - yoffset;

        let pixel = self
            .transform
            .apply_inverse(Point3::new(world_x, world_y, -1.0))?;
        let origin = self.transform.apply_inverse(Point3::new(0.0, 0.0, 0.0))?;
        let direction = (pixel - origin).normalize();

        Ok(Ray::new(origin, direction))
    }

    pub fn render(&self, world: &World) -> Result<Canvas, RenderError> {
        let mut image = Canvas::new(self.hsize, self.vsize);

        for y in 0..self.vsize {
            for x in 0..self.hsize {
                let ray = self.ray_for_pixel(x, y)?;
                let color = world.color_at(&ray, MAX_RECURSION_DEPTH)?;
                image.write_pixel(x, y, color)?;
            }
        }

        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;
    use crate::{geometry::transforms::Axis, scene::world::World};
    use approx::assert_abs_diff_eq;
    use nalgebra::{Translation3, Vector3};

    #[test]
    fn test_constructing_a_camera() {
        let hsize = 160;
        let vsize = 120;
        let field_of_view = std::f64::consts::PI / 2.0;
        let transform = Transform::identity();
        let camera = Camera::new(hsize, vsize, field_of_view, transform);

        assert_eq!(camera.hsize, hsize);
        assert_eq!(camera.vsize, vsize);
        assert_eq!(camera.field_of_view, field_of_view);
        assert_eq!(camera.transform, transform);
    }

    #[test]
    fn test_pixel_size_for_horizontal_canvas() {
        let camera = Camera::new(200, 125, std::f64::consts::PI / 2.0, Transform::identity());

        assert_abs_diff_eq!(camera.pixel_size, 0.01);
    }

    #[test]
    fn test_pixel_size_for_vertical_canvas() {
        let camera = Camera::new(125, 200, std::f64::consts::PI / 2.0, Transform::identity());

        assert_abs_diff_eq!(camera.pixel_size, 0.01);
    }

    #[test]
    fn constructing_a_ray_through_the_center_of_the_canvas() {
        let camera = Camera::new(201, 101, std::f64::consts::PI / 2.0, Transform::identity());
        let ray = camera.ray_for_pixel(100, 50).unwrap();

        assert_abs_diff_eq!(ray.origin.x, 0.0);
        assert_abs_diff_eq!(ray.origin.y, 0.0);
        assert_abs_diff_eq!(ray.origin.z, 0.0);
        assert_abs_diff_eq!(ray.direction.x, 0.0);
        assert_abs_diff_eq!(ray.direction.y, 0.0);
        assert_abs_diff_eq!(ray.direction.z, -1.0);
    }

    #[test]
    fn constructing_a_ray_through_a_corner_of_the_canvas() {
        let camera = Camera::new(201, 101, std::f64::consts::PI / 2.0, Transform::identity());
        let ray = camera.ray_for_pixel(0, 0).unwrap();

        assert_abs_diff_eq!(ray.origin.x, 0.0);
        assert_abs_diff_eq!(ray.origin.y, 0.0);
        assert_abs_diff_eq!(ray.origin.z, 0.0);
        assert_abs_diff_eq!(ray.direction.x, 0.66519, epsilon = 1e-5);
        assert_abs_diff_eq!(ray.direction.y, 0.33259, epsilon = 1e-5);
        assert_abs_diff_eq!(ray.direction.z, -0.66851, epsilon = 1e-5);
    }

    #[test]
    fn constructing_a_ray_when_the_camera_is_transformed() {
        let transform = Transform::sequence([
            Transform::translation(Translation3::new(0.0, -2.0, 5.0)),
            Transform::rotation(Axis::Y, std::f64::consts::FRAC_PI_4),
        ]);
        let camera = Camera::new(201, 101, std::f64::consts::PI / 2.0, transform);
        let ray = camera.ray_for_pixel(100, 50).unwrap();

        assert_abs_diff_eq!(ray.origin.x, 0.0);
        assert_abs_diff_eq!(ray.origin.y, 2.0);
        assert_abs_diff_eq!(ray.origin.z, -5.0);
        assert_abs_diff_eq!(ray.direction.x, f64::consts::SQRT_2 / 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(ray.direction.y, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(ray.direction.z, -f64::consts::SQRT_2 / 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_rendering_a_world_with_a_camera() {
        let world = World::default();
        let transform = World::view_transform(
            Point3::new(0.0, 0.0, -5.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let camera = Camera::new(11, 11, std::f64::consts::PI / 2.0, transform);
        let image = camera.render(&world).unwrap();

        let pixel_color = image.pixels[5 * image.width + 5];
        assert_abs_diff_eq!(pixel_color.r(), 0.38066, epsilon = 1e-5);
        assert_abs_diff_eq!(pixel_color.g(), 0.47583, epsilon = 1e-5);
        assert_abs_diff_eq!(pixel_color.b(), 0.2855, epsilon = 1e-5);
    }
}
