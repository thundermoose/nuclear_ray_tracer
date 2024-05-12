#include <ray_integral/ray_integral.hh>
#include <thrust/transform.h>

nucray::ray_integral::ray_integral(int num_steps, float step_length, nucray::onebody_density density)
    : num_steps(num_steps), step_length(step_length), density(density) {
}

thrust::device_vector<float>
    nucray::ray_integral::integrate(thrust::device_vector<ray> &rays) {
        thrust::device_vector<float> amplitudes(rays.size());
        thrust::transform(rays.begin(), rays.end(), amplitudes.begin(), *this);
        return amplitudes;
}

__device__ float nucray::ray_integral::operator()(nucray::ray &r) const {
        float previous = 0.0f;
        float parameter = 0;
        float accumulator = 0.0f;
        for (int i = 0; i <= num_steps; i++) {
                float current = density(r.get_point(parameter));
                if (i > 0) {
                        accumulator += 0.5*(current+previous)*step_length;
                }
                parameter+=step_length;
                previous = current;
        }
        return accumulator;
}
