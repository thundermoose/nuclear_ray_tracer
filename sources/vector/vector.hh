#ifndef __VECTOR__
#define __VECTOR__

const float tollerance = 1e-5;
namespace nucray {
class vector {
      public:
        float x, y, z;
        __host__ __device__ vector() : x(0.0f), y(0.0f), z(0.0f) {
        }

        __host__ __device__ vector(float x, float y, float z)
            : x(x), y(y), z(z) {
        }

        __host__ __device__ vector(const vector &other)
            : x(other.x), y(other.y), z(other.z) {
        }

        __host__ __device__ vector(vector &&other)
            : x(other.x), y(other.y), z(other.z) {
        }

        __host__ __device__ void operator=(const vector &origin) {
                x = origin.x;
                y = origin.y;
                z = origin.z;
        }

        __host__ __device__ vector operator+(const vector right_term) const {
                return vector(
                    x + right_term.x, y + right_term.y, z + right_term.z);
        }

        __host__ __device__ vector operator-(const vector right_term) const {
                return vector(
                    x - right_term.x, y - right_term.y, z - right_term.z);
        }

        __host__ __device__ float operator*(const vector right_term) const {
                return x * right_term.x + y * right_term.y + z * right_term.z;
        }

        __host__ __device__ vector operator*(const float scale) const {
                return vector(x * scale, y * scale, z * scale);
        }

        __host__ __device__ vector cros(const vector right_term) const {
                return vector(y * right_term.z - z * right_term.y,
                              z * right_term.x - x * right_term.z,
                              x * right_term.y - y * right_term.x);
        }

        __host__ __device__ float norm() const {
                return sqrtf(x * x + y * y + z * z);
        }

        __host__ __device__ float norm_squared() const {
                return x * x + y * y + z * z;
        }

        __host__ __device__ float theta() const {
                return acosf(z/norm());
        }

        __host__ __device__ float phi() const {
                float rho = sqrtf(x*x + y*y);
                if (y > 0)
                        return acosf(x/rho);
                else
                        return -acosf(x/rho);
        }

        __host__ __device__ void normalize() {
                auto distance = std::sqrt((*this) * (*this));
                x /= distance;
                y /= distance;
                z /= distance;
        }

        __host__ __device__ bool operator==(const vector right_term) {
                auto diff = (*this) - right_term;
                return fabs(diff.x) < tollerance && fabs(diff.y) < tollerance &&
                       fabs(diff.z) < tollerance;
        }

        __host__ __device__ ~vector() {
        }
};
} // namespace nucray
#endif
