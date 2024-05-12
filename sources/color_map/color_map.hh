#ifndef __COLOR_MAP__
#define __COLOR_MAP__

#include <color_map/color.hh>
#include <memory>
#include <thrust/device_vector.h>
namespace nucray {
class color_map {
      public:
        color_map() = default;
        ~color_map() = default;
        virtual thrust::device_vector<color>
            apply(thrust::device_vector<float> &amplitudes) = 0;
};
class color_map_factory {
      public:
        color_map_factory() = default;
        virtual ~color_map_factory() = default;
        virtual std::shared_ptr<color_map> create_color_map() = 0;
};
} // namespace nucray

#endif
