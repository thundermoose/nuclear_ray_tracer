#include <blixttest/test.hh>
#include <camera/camera.hh>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <vector>

nucray::camera::camera(nucray::vector origin,
                       nucray::vector side,
                       nucray::vector up,
                       float focal_distance,
                       size_t width,
                       size_t height)
    : origin(origin), forward(up.cros(side)), up(up), side(side),
      focal_distance(focal_distance), width(width), height(height) {
}

thrust::device_vector<nucray::ray> nucray::camera::get_rays() {
        thrust::device_vector<size_t> indices(this->width * this->height);
        thrust::sequence(indices.begin(), indices.end());
        thrust::device_vector<ray> rays(this->width * this->height);
        thrust::transform(indices.begin(), indices.end(), rays.begin(), *this);
        return rays;
}

__device__ nucray::ray nucray::camera::operator()(size_t &ray_index) const {
        size_t row = ray_index / this->width;
        size_t column = ray_index % this->width;
        nucray::vector lower_left_corner = this->origin - this->side - this->up;
        nucray::vector pos = lower_left_corner +
                             this->side * (2.0f * (float)column / this->width) +
                             this->up * (2.0f * (float)row / this->height);
        nucray::vector dir = (pos - (origin - forward * focal_distance));
        dir.normalize();
        return nucray::ray(pos, dir);
}

new_test(creating_rays) {
        nucray::camera cam(nucray::vector(0, 0, 0),
                           nucray::vector(1, 0, 0),
                           nucray::vector(0, 1, 0),
                           1.0f,
                           4,
                           4);
        auto rays_device = cam.get_rays();
        std::vector<nucray::ray> rays_host(rays_device.size());
        thrust::copy(rays_device.begin(), rays_device.end(), rays_host.begin());
        for (auto &ray : rays_host) {
                std::cout << "ray origin: " << ray.start_position.x << " "
                          << ray.start_position.y << " " << ray.start_position.z
                          << " ray direction:" << ray.direction.x << " "
                          << ray.direction.y << " "
                          << " " << ray.direction.z << std::endl;
        }
}
