#ifndef __COLOR__
#define __COLOR__

#include <cstdint>

namespace nucray {
union color {
        uint32_t color_code;
        struct {
                uint8_t red, green, blue, alpha;
        } color_channals;
};
} // namespace nucray
#endif
