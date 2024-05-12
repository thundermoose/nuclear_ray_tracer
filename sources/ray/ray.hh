#ifndef __RAY__
#define __RAY__

#include <vector/vector.hh>

namespace nucray {
class ray {
      public:
        vector start_position, direction;

        __host__ __device__ ray() {
        }
        __host__ __device__ ray(vector start_position, vector direction)
            : start_position(start_position), direction(direction) {
        }
        __device__ vector get_point(float parameter) const {
                return vector(start_position.x + direction.x * parameter,
                              start_position.y + direction.y * parameter,
                              start_position.z + direction.z * parameter);
        }
};
} // namespace nucray

#endif
