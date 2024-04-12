#ifndef __RAY__
#define __RAY__

#include <vector/vector.hh>

namespace nucray {
class ray {
        vector start_position, direction;

      public:
        __host__ __device__ ray() = default;
        __host__ __device__ ray(vector start_position, vector direction);
};
} // namespace nucray

#endif
