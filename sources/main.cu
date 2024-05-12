#include <blixttest/test.hh>
#include <camera/camera.hh>
#include <color_map/color_map_picker.hh>
#include <harmonic_state/harmonic_state.hh>
#include <onebody_density/onebody_density.hh>
#include <ppm_writer/ppm_writer.hh>
#include <ray_integral/ray_integral.hh>
#include <sstream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <vector/vector.hh>
#include <vector>

const size_t width = 800;
const size_t height = 800;

float min(thrust::device_vector<float> &array) {
        return *thrust::min_element(array.begin(), array.end());
}

float max(thrust::device_vector<float> &array) {
        return *thrust::max_element(array.begin(), array.end());
}

int main() {
        nucray::camera camera(nucray::vector(0.0f, 0.0f, 2.0f),
                              nucray::vector(1.0f, 0.0f, 0.0f),
                              nucray::vector(0.0f, 1.0f, 0.0f),
                              0.5,
                              width,
                              height);
        for (int l = 0; l <= 5; l++) {
                for (int m = -l; m <= l; m++) {
                        nucray::onebody_density density(
                            1.0f,
                            nucray::harmonic_state(2, l, m, 5.0f),
                            nucray::harmonic_state(2, l, m, 5.0f));
                        nucray::ray_integral integral(100, 0.1, density);
                        auto rays = camera.get_rays();
                        auto amplitudes = integral.integrate(rays);

                        nucray::color_map_picker pick_color(min(amplitudes),
                                                            max(amplitudes));
                        auto color_map = pick_color.get_color_map("blue-green-yellow-red");
                        auto pixels = color_map->apply(amplitudes);
                        std::vector<nucray::color> host_pixels(pixels.size());
                        thrust::copy(
                            pixels.begin(), pixels.end(), host_pixels.begin());
                        nucray::ppm_writer ppm_writer;
                        ppm_writer.set_pixels(width, height, host_pixels);
                        std::stringstream strs;
                        strs << "images/harmonic_l=" << l << "_m=" << m
                             << "_image.ppm";
                        ppm_writer.write(strs.str());
                }
        }
}
