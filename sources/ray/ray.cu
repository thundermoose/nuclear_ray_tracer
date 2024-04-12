#include <curand.h>
#include <ray/ray.hh>

__host__ __device__ nucray::ray::ray(nucray::vector start_position,
                                     nucray::vector direction)
    : start_position(start_position), direction(direction) {}
