#ifndef __CAMERA__
#define __CAMERA__

#include <ray/ray.hh>
#include <thrust/device_vector.h>
#include <vector/vector.hh>
#include <functional>

namespace nucray {
class camera {
        vector origin, forward, up, side;
        float focal_distance;
        size_t width, height;

      public:
        camera(vector origin, vector side, vector up, float focal_distance,
               size_t width, size_t height);

        thrust::device_vector<ray> get_rays();
        __device__
       ray operator ()(size_t &ray_index) const;
      private:
};
} // namespace nucray

#endif
