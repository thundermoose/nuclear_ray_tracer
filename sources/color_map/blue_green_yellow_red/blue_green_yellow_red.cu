#include <color_map/blue_green_yellow_red/blue_green_yellow_red.hh>
#include <cstdint>

class blue_green_yellow_red_functor{
        float minimum, maximum;

      public:
        __host__ __device__ blue_green_yellow_red_functor(float minimum, float maximum)
            : minimum(minimum), maximum(maximum) {
        }
        __device__ nucray::color operator()(float &value) const {
                float intensity =
                    (value > minimum
                        ? (value < maximum ? value : maximum) - minimum
                        : 0.0f)/(maximum - minimum);
                float blue_amount = 0;
                float green_amount = 0;
                float yellow_amount = 0;
                float red_amount = 0;
                if (intensity < 1.0/3) {
                        blue_amount = (3*(1.0/3 - intensity));
                        green_amount = (3*(intensity));
                } else if (intensity < 2.0/3) {
                        green_amount = (3*(2.0/3-intensity));
                        yellow_amount = (3*(intensity - 1.0/3));
                } else {
                        yellow_amount = (3*(1.0-intensity));
                        red_amount = (3*(intensity-2.0/3));
                }
                red_amount += yellow_amount;
                green_amount += yellow_amount;

                nucray::color result = {.color_channals = {
                                    .red = (uint8_t)(255*sqrtf(red_amount)),
                                    .green = (uint8_t)(255*sqrtf(green_amount)),
                                    .blue = (uint8_t)(255*sqrtf(blue_amount)),
                                    .alpha = 0xff,
                                }};
                return result;
        }
};

nucray::blue_green_yellow_red_factory::blue_green_yellow_red_factory(float minimum, float maximum)
    : minimum(minimum), maximum(maximum) {
}

thrust::device_vector<nucray::color>
    nucray::blue_green_yellow_red::apply(thrust::device_vector<float> &amplitudes) {
        thrust::device_vector<nucray::color> colors(amplitudes.size());
        blue_green_yellow_red_functor functor(minimum, maximum);
        thrust::transform(
            amplitudes.begin(), amplitudes.end(), colors.begin(), functor);
        return colors;
}

std::shared_ptr<nucray::color_map>
    nucray::blue_green_yellow_red_factory::create_color_map() {
        return std::make_shared<blue_green_yellow_red>(minimum, maximum);
}
