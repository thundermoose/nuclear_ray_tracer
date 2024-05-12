#ifndef __ONEBODY_DENSITY__
#define __ONEBODY_DENSITY__

#include <harmonic_state/harmonic_state.hh>
#include <vector/vector.hh>

namespace nucray {
class onebody_density {
        float amplitude;
        harmonic_state bra, ket;

      public:
        onebody_density(float amplitude, harmonic_state bra, harmonic_state ket)
            : amplitude(amplitude), bra(bra), ket(ket) {
        }
        __device__ float operator()(vector position) const {
                return amplitude * (bra.wave_function(position) *
                       ket.wave_function(position)).real();
        }
};
} // namespace nucray

#endif
