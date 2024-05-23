#ifndef __SPHERICAL_HARMONICS__
#define __SPHERICAL_HARMONICS__

#include <legendre/legendre.hh>
#include <thrust/complex.h>

namespace nucray {
class spherical_harmonics {
        int l, m;
        float normalization;
        legendre L;
      public:
        spherical_harmonics(int l, int m);
        __device__ thrust::complex<float> operator()(float theta,
                                                     float phi) const {
                return normalization * L(cosf(theta)) *
                       exp(thrust::complex<float>(0.0f, 1.0f) * m * phi);
        }
};
} // namespace nucray
#endif
