#ifndef __ADFLOAT__
#define __ADFLOAT__

#include <cinttypes>
#include <cmath>
namespace nucray {
struct adfloat {
        float value, derivative;
        __host__ __device__ adfloat(float value, float derivative)
            : value(value), derivative(derivative) {
        }
        __host__ __device__ adfloat operator+(const adfloat other) {
                return adfloat(value + other.value,
                               derivative + other.derivative);
        }
        __host__ __device__ adfloat operator-(const adfloat other) {
                return adfloat(value - other.value,
                               derivative - other.derivative);
        }
        __host__ __device__ adfloat operator*(const adfloat other) {
                return adfloat(value * other.value,
                               value * other.derivative +
                                   derivative * other.value);
        }
        __host__ __device__ adfloat operator/(const adfloat other) {
                return adfloat(
                    value / other.value,
                    (-value * other.derivative + derivative * other.value) /
                        (other.value * other.value));
        }
        __host__ __device__ adfloat exp() {
                return adfloat(expf(value), derivative * expf(value));
        }
        __host__ __device__ adfloat sin() {
                return adfloat(sinf(value), derivative * cosf(value));
        }
        __host__ __device__ adfloat cos() {
                return adfloat(cosf(value), -derivative * sinf(value));
        }
        __host__ __device__ adfloat sqrt() {
                return adfloat(sqrtf(value), derivative / sqrtf(value));
        }
        __host__ __device__ adfloat pow(float exponent) {
                return adfloat(powf(value, exponent), fabs(exponent) < 1e-4 ? 0.0f : exponent*powf(value, exponent-1.0f));
        }
        __host__ __device__ adfloat neg() {
                return adfloat(-value, -derivative);
        }
        __host__ __device__ adfloat asin() {
                return adfloat(asinf(value),derivative/sqrtf(1-value*value));
        }
};
} // namespace nucray
#endif
