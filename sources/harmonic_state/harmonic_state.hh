#ifndef __HARMONIC_STATE__
#define __HARMONIC_STATE__

#include <laguerre/laguerre.hh>
#include <vector/vector.hh>
#include <thrust/complex.h>
#include <spherical_harmonics/spherical_harmonics.hh>
namespace nucray {
class harmonic_state {
        int n, l, m;
        float frequency;
        float mass;
        float normalization;
        float nu;
        spherical_harmonics Y;
      public:
        harmonic_state(
            int n, int l, int m, float frequency = 20.0f, float mass = 1.0f);
        __device__ thrust::complex<float> wave_function(vector position) const {
                float radius = position.norm();
                float theta = position.theta();
                float phi = position.phi();

                return normalization * powf(radius, l) *
                       expf(-nu * radius * radius) *
                       laguerre(n, l + 0.5f, 2 * nu * radius * radius) *
                       Y(theta, phi);
        }
};
} // namespace nucray

#endif
