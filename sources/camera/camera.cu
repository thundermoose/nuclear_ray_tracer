#include "ray/ray.hh"
#include "vector/vector.hh"
#include <camera/camera.hh>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

nucray::camera::camera(nucray::vector origin, nucray::vector side,
                       nucray::vector up, float focal_distance, size_t width,
                       size_t height)
    : origin(origin), forward(up.cros(side)), up(up), side(side),
      focal_distance(focal_distance), width(width), height(height) {}

thrust::device_vector<nucray::ray> nucray::camera::get_rays() {
        thrust::device_vector<size_t> indices(this->width * this->height);
        thrust::sequence(indices.begin(), indices.end());
        thrust::device_vector<ray> rays;
        thrust::transform(indices.begin(), indices.end(), rays.begin(), *this);
}

__device__ nucray::ray nucray::camera::operator()(size_t ray_index) {
        size_t row = ray_index / this->width;
        size_t column = ray_index % this->width;
        auto lower_left_corner = this->origin - this->side - this->up;
        auto pos = lower_left_corner +this->side*(2.0*(float)column/this->width) + this->up*(2.0*(float)row/this->height);
        auto dir = (pos - (origin - forward*focal_distance));
        dir.normalize();
        return nucray::ray(pos, dir);
}
