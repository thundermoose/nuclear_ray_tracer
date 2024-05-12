#ifndef __LAGUERRE__
#define __LAGUERRE__

__device__ static inline 
float laguerre(int n, float alpha, float x) {
        float previous = 0.0f;
        float current = 0.0f;
        previous = 1;
        current = 1 + alpha - x;
        for (int k = 2; k <= n; k++) {
                float next = ((2*k - 1 + alpha - x)*current - (k - 1 + alpha)*previous)/k;
                previous = current;
                current = next;
        }
        return current;
}

#endif
