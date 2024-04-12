#include "vector.hh"
#include <blixttest/test.hh>
#include <cmath>

const float tollerance = 1e-5;

__host__ __device__ nucray::vector::vector(float x, float y, float z)
    : x(x), y(y), z(z) {}

__host__ __device__ nucray::vector::operator=(const vector &origin) {
        x = origin.x;
        y = origin.y;
        z = origin.z;
}

__host__ __device__ nucray::vector
nucray::vector::operator+(const vector &right_term) {
        return vector(x + right_term.x, y + right_term.y, z + right_term.z);
}

__host__ __device__ nucray::vector
nucray::vector::operator-(const vector &right_term) {
        return vector(x - right_term.x, y - right_term.y, z - right_term.z);
}

__host__ __device__ float nucray::vector::operator*(const vector &right_term) {
        return x * right_term.x + y * right_term.y + z * right_term.z;
}

__host__ __device__ nucray::vector
nucray::vector::operator*(const float scale) {
        return nucray::vector(x * scale, y * scale, z * scale);
}

__host__ __device__ nucray::vector
nucray::vector::cros(const vector &right_term) {
        return nucray::vector(y * right_term.z - z * right_term.y,
                              z * right_term.x - x * right_term.z,
                              x * right_term.y - y * right_term.x);
}

__host__ __device__ void nucray::vector::normalize() {
        auto distance = std::sqrt((*this) * (*this));
        x /= distance;
        y /= distance;
        z /= distance;
}

__host__ __device__ bool nucray::vector::operator==(const vector &right_term) {
        auto diff = (*this) - right_term;
        return fabs(diff.x) < tollerance && fabs(diff.y) < tollerance &&
               fabs(diff.z) < tollerance;
}

new_test(adding_vectors) {
        assert_that(nucray::vector(3, 4, 5) + nucray::vector(3, 2, 1) ==
                    nucray::vector(6, 6, 6));
}

new_test(subtracting_vectors) {
        assert_that(nucray::vector(3, 4, 5) - nucray::vector(1, 2, 3) ==
                    nucray::vector(2, 2, 2));
}

new_test(scalarproduct) {
        assert_that(fabs(nucray::vector(1, 1, 1) * nucray::vector(1, 1, 1) -
                         3) < tollerance);
}

new_test(scale_vector) {
        assert_that(nucray::vector(1, 2, 3) * 2 == nucray::vector(2, 4, 6));
}

new_test(crossproduct) {
        assert_that(nucray::vector(1, 0, 0).cros(nucray::vector(0, 1, 0)) ==
                    nucray::vector(0, 0, 1));
        assert_that(nucray::vector(0, 1, 0).cros(nucray::vector(0, 0, 1)) ==
                    nucray::vector(1, 0, 0));
        assert_that(nucray::vector(0, 0, 1).cros(nucray::vector(1, 0, 0)) ==
                    nucray::vector(0, 1, 0));
        assert_that(nucray::vector(0, 1, 0).cros(nucray::vector(1, 0, 0)) ==
                    nucray::vector(0, 0, -1));
        assert_that(nucray::vector(0, 0, 1).cros(nucray::vector(0, 1, 0)) ==
                    nucray::vector(-1, 0, 0));
        assert_that(nucray::vector(1, 0, 0).cros(nucray::vector(0, 0, 1)) ==
                    nucray::vector(0, -1, 0));
}
