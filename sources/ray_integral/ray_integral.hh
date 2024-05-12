#ifndef __RAY_INTEGRAL__
#define __RAY_INTEGRAL__

#include <ray/ray.hh>
#include <onebody_density/onebody_density.hh>
#include <thrust/device_vector.h>

namespace nucray {
class ray_integral {
        int num_steps;
        float step_length;
        onebody_density density;
      public:
        __host__ ray_integral(int num_steps, float step_length, onebody_density density);
        thrust::device_vector<float>
            integrate(thrust::device_vector<ray> &rays);
        __device__ float operator() (ray &r) const;
                
};
} // namespace nucray

#endif
