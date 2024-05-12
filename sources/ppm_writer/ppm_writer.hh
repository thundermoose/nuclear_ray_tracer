#ifndef __PPM_WRITER__
#define __PPM_WRITER__

#include <cstdio>
#include <string>
#include <vector>
#include <color_map/color.hh>
namespace nucray {
class ppm_writer {
        size_t width, height;
        std::vector<nucray::color> pixels;
      public:
        ppm_writer();

        void set_pixels(size_t width, size_t height, std::vector<nucray::color> pixels);
        void write(std::string filename);
};
} // namespace nucray

#endif
