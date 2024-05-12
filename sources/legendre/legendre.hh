#ifndef __LEGENDRE__
#define __LEGENDRE__

__device__ static inline float legendre(int l, int m, float x) {
        if (l == 0)
                return 1.0;
        float previous = 1.0f;
        float current = x;
        if (m > 0) {
                for (int mm = 0; mm < m - 1; mm++) {
                        float next1 =
                            (((2 * mm + 3) * x - (2 * mm + 2)) * current -
                             (2 * mm + 1) * previous) /
                            sqrtf(1 - x * x);
                        float intermediat = (x * (2 * mm + 3) * current -
                                             (2 * mm + 1) * previous) /
                                            2;
                        float next2 =
                            (((2 * mm + 5) * x - (2 * mm + 3)) * intermediat -
                             (2 * mm + 2) * current) /
                            sqrtf(1 - x * x);
                        previous = next1;
                        current = next2;
                }
                // Now current = P_(m+1)^m and previous = P_(m)^m
        } else if (m < 0) {
                for (int mm = 0; mm < m - 1; mm++) {
                        float next1 =
                            (((2 * mm + 3) * x - (2 * mm + 2)) * current -
                             (2 * mm + 1) * previous) /
                            (sqrtf(1 - x * x)*(2*mm+3)*(2*mm+2));
                        float intermediat = (x * (2 * mm + 3) * current -
                                             (2 * mm + 1) * previous) /
                                            2;
                        float next2 =
                            (((2 * mm + 5) * x - (2 * mm + 3)) * intermediat -
                             (2 * mm + 2) * current) /
                            sqrtf(1 - x * x);
                        previous = next1;
                        current = next2;
                }
                // Now current = P_(|m|+1)^(-m) and previous = P_(|m|)^(-m)
        }

        for (int k = abs(m)+2; k <=l; k++) {
                float next = (2*k - 1)*x*current -(k+m - 1)*previous;
                previous = current;
                current = next;
        }

        return current;
}

#endif
