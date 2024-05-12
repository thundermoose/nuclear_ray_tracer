#ifndef __LEGENDRE__
#define __LEGENDRE__

#include <iostream>

/**
 * @name legendre
 * @breif This function compute the associated Legendre polynomials, not to be
 *        confused with the Legendre polynomials.
 */
__device__ __host__ static inline float legendre(int l, int m, float x) {
        if (l == 0)
                return 1.0;
        float previous = 1.0f;
        float current = x;
        if (m > 0) {
                for (int mm = 0; mm < m; mm++) {
                        previous = -(2 * mm + 1) * sqrtf(1 - x * x) * previous;
                }
                current = x * (2 * m + 1) * previous;
        } else if (m < 0) {
                for (int mm = 0; mm < -m; mm++) {
                        previous = sqrt(1 - x * x) * previous / (2 * mm + 2);
                }
                current = x * (-2 * m + 1) * previous;
        }
        for (int k = abs(m) + 2; k <= l; k++) {
                float next =
                    ((2 * k - 1) * x * current - (k + m - 1) * previous) /
                    (k - m);
                previous = current;
                current = next;
        }
        if (l == abs(m))
                return previous;
        else
                return current;
}
#endif
