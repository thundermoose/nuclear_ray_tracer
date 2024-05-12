#ifndef __BLUE_GREEN_YELLOW_RED__
#define __BLUE_GREEN_YELLOW_RED__

#include <color_map/color_map.hh>
#include <cstdint>
#include <memory>

namespace nucray {
class blue_green_yellow_red : public color_map {
        float minimum, maximum;

      public:
        blue_green_yellow_red(float minimum, float maximum)
            : minimum(minimum), maximum(maximum) {
        }

        thrust::device_vector<color>
            apply(thrust::device_vector<float> &amplitudes) override;

};
class blue_green_yellow_red_factory : public color_map_factory {
        float minimum, maximum;

      public:
        blue_green_yellow_red_factory(float minimum, float maximum);
        std::shared_ptr<color_map> create_color_map();
};
} // namespace nucray

#endif
