#include <cmath>
#include <harmonic_state/harmonic_state.hh>
#include <combinatorics/combinatorics.hh>

const float hbar = 1.0f;



nucray::harmonic_state::harmonic_state(
    int n, int l, int m, float frequency, float mass)
    : n(n), l(l), m(m), frequency(frequency), mass(mass), Y(l,m) {
        nu = mass * frequency / (2 * hbar);
        normalization =
            sqrtf(sqrt(2 * nu * nu * nu / M_PI) *
                  ((1 << (n + 2 * l + 3)) * factorial(n) * powf(nu, (float)l)) /
                  double_factorial(2 * n + 2 * l + 1));
}
