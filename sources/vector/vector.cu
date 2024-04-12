#include "vector.hh"
#include <blixttest/test.hh>
#include <cmath>

const float tollerance = 1e-5;

__host__ __device__ nucray::vector::vector(float x, float y, float z)
    : x(x), y(y), z(z) {}

__host__ __device__ nucray::vector
nucray::vector::operator+(const vector &right_term) {
        return vector(x + right_term.x, y + right_term.y, z + right_term.z);
}

__host__ __device__ nucray::vector
nucray::vector::operator-(const vector &right_term) {
        return vector(x - right_term.x, y - right_term.y, z - right_term.z);
}

__host__ __device__ float nucray::vector::operator*(const vector &right_term) {
        return x*right_term.x + y*right_term.y + z*right_term.z;
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
        assert_that(fabs(nucray::vector(1,1,1)*nucray::vector(1,1,1) - 3) < tollerance);
}
