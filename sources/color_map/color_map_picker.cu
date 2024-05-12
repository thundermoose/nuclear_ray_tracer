#include <color_map/color_map.hh>
#include <algorithm>
#include <color_map/blue_to_red/blue_to_red.hh>
#include <color_map/color_map_picker.hh>
#include <memory>
#include <sstream>
#include <stdexcept>

nucray::color_map_picker::color_map_picker(float minimum, float maximum)
    : minimum(minimum), maximum(maximum) {
        available_color_maps["blue-red"] =
            std::make_unique<nucray::blue_to_red_factory>(minimum, maximum);
}

std::vector<std::string>
    nucray::color_map_picker::get_available_color_map_names() {
        std::vector<std::string> keys;
        for (auto &key_value: available_color_maps) {
            keys.push_back(key_value.first);
        }
        return keys;
}

std::shared_ptr<nucray::color_map> nucray::color_map_picker::get_color_map(std::string name) {
    try {
        return available_color_maps.at(name)->create_color_map();
    } catch (std::out_of_range &e){
        std::stringstream error_message;
        error_message << "There is no color map avialble with name: " << name;
        throw std::out_of_range(error_message.str().c_str());
    }
}
