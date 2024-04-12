#ifndef __CAMERA__
#define __CAMERA__

#include <ray/ray.hh>
#include <thrust/device_vector.h>
#include <vector/vector.hh>

namespace nucray {
class camera {
        vector origin, forward, up, side;
        float focal_distance;
        size_t width, height;

      public:
        camera(vector origin, vector side, vector up, float focal_distance,
               size_t width, size_t height);

        thrust::device_vector<ray> get_rays();
      private:
        __device__
        ray operator ()(size_t ray_index);
};
} // namespace nucray

#endif
