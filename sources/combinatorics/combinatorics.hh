#ifndef __COMBINATORICS__
#define __COMBINATORICS__

static inline float factorial(int n) {
        float accumulator = 0;
        for (int i = 2; i <= n; i++)
                accumulator += logf(i);
        return expf(accumulator);
}

static inline float double_factorial(int n) {
        float accumulator = 0;
        for (int i = (n & 1) + 2; i <= n; i += 2)
                accumulator += logf(i);
        return expf(accumulator);
}

#endif
