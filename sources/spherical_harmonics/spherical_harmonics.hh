#ifndef __SPHERICAL_HARMONICS__
#define __SPHERICAL_HARMONICS__

#include <thrust/complex.h>
#include <legendre/legendre.hh>

namespace nucray {
class spherical_harmonics {
        int l, m;
        float normalization;

      public:
        spherical_harmonics(int l, int m);
        __device__ thrust::complex<float> operator()(float theta, float phi) const {
                return normalization*legendre(l,m,cosf(theta))*exp(thrust::complex<float>(0.0f,1.0f)*m*phi);
        }
};
} // namespace nucray
#endif
