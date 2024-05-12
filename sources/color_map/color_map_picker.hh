#ifndef __COLOR_MAP_PICKER__
#define __COLOR_MAP_PICKER__

#include <color_map/color_map.hh>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace nucray {
class color_map_picker {
        float minimum, maximum;
        std::map<std::string, std::unique_ptr<color_map_factory>>
            available_color_maps;

      public:
        color_map_picker(float minimum, float maximum);
        std::shared_ptr<color_map> get_color_map(std::string name);
        std::vector<std::string> get_available_color_map_names();
};
} // namespace nucray
#endif
