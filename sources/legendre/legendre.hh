#ifndef __LEGENDRE__
#define __LEGENDRE__

#include <cmath>
#include <iostream>

namespace nucray {
class legendre {
        int l, m;
        float prefactor;

      public:
        legendre(int l, int m) : l(l), m(m) {
                prefactor= 1.0;
                int start = m > 0 ? 1 : 2;
                for (int mm = 0; mm < abs(m); mm++) {
                        prefactor= (2 * mm + start) * prefactor;
                }
        }
        /**
         * @name legendre
         * @breif This function compute the associated Legendre polynomials, not
         * to be confused with the Legendre polynomials.
         */
        __device__ __host__ float operator()(float x) const {
                if (l == 0)
                        return 1.0;
                float previous = prefactor;
                float current = x;
                if (m > 0) {
                        previous *= powf(-sqrtf(1 - x * x), m);
                        current = x * (2 * m + 1) * previous;
                } else if (m < 0) {
                        previous = powf(sqrtf(1 - x * x), -m) / previous;
                        current = x * (-2 * m + 1) * previous;
                }
                for (int k = abs(m) + 2; k <= l; k++) {
                        float next = ((2 * k - 1) * x * current -
                                      (k + m - 1) * previous) /
                                     (k - m);
                        previous = current;
                        current = next;
                }
                if (l == abs(m))
                        return previous;
                else
                        return current;
        }
};
} // namespace nucray
#endif
