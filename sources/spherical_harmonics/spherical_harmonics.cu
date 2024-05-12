#include <spherical_harmonics/spherical_harmonics.hh>
#include <combinatorics/combinatorics.hh>

nucray::spherical_harmonics::spherical_harmonics(int l, int m): l(l), m(m) {
        normalization = sqrtf((2*l + 1)/(4*M_PI) * factorial(l - m)/(factorial(l+m)));
}
