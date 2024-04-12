#ifndef __VECTOR__
#define __VECTOR__
namespace nucray {
class vector {
        float x, y, z;

      public:
        __host__ __device__ vector() = default;
        __host__ __device__ vector(float x, float y, float z);

        __host__ __device__ vector(const vector &) = default;

        __host__ __device__ vector(vector &&) = default;

        __host__ __device__ void operator = (const vector &origin);

        __host__ __device__ vector operator+(const vector &right_term);

        __host__ __device__ vector operator-(const vector &right_term);

        __host__ __device__ float operator*(const vector &right_term);

        __host__ __device__ vector operator*(const float scale);

        __host__ __device__ vector cros(const vector &right_term);

        __host__ __device__ bool operator==(const vector &right_term);

        __host__ __device__ void normalize();
};
} // namespace nucray
#endif
