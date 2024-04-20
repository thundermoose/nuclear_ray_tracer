#ifndef __RAY__
#define __RAY__

#include <vector/vector.hh>

namespace nucray {
class ray {
      public:
        vector start_position, direction;

        __host__ __device__ ray() {}
        __host__ __device__ ray(vector start_position, vector direction) :
                start_position(start_position), direction(direction) {}
};
} // namespace nucray

#endif
