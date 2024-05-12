#include <color_map/blue_to_red/blue_to_red.hh>
#include <memory>
#include <thrust/transform.h>

class blue_to_red_functor {
        float minimum, maximum;

      public:
        __host__ __device__ blue_to_red_functor(float minimum, float maximum)
            : minimum(minimum), maximum(maximum) {
        }
        __device__ nucray::color operator()(float &value) const {
                float intensity =
                    (value > minimum
                        ? (value < maximum ? value : maximum) - minimum
                        : 0.0f)/(maximum - minimum);
                float blue_amount = 255*sqrtf(1.0f - intensity);
                float red_amount = 255*sqrtf(intensity);
                nucray::color result = {.color_channals = {
                                    .red = (uint8_t)(red_amount),
                                    .green = 0,
                                    .blue = (uint8_t)(blue_amount),
                                    .alpha = 0xff,
                                }};
                return result;
        }
};

nucray::blue_to_red_factory::blue_to_red_factory(float minimum, float maximum)
    : minimum(minimum), maximum(maximum) {
}

thrust::device_vector<nucray::color>
    nucray::blue_to_red::apply(thrust::device_vector<float> &amplitudes) {
        thrust::device_vector<nucray::color> colors(amplitudes.size());
        blue_to_red_functor functor(minimum, maximum);
        thrust::transform(
            amplitudes.begin(), amplitudes.end(), colors.begin(), functor);
        return colors;
}

std::shared_ptr<nucray::color_map>
    nucray::blue_to_red_factory::create_color_map() {
        return std::make_shared<blue_to_red>(minimum, maximum);
}
