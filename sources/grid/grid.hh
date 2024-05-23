#ifndef __GRID__
#define __GRID__

#include <thrust/device_vector.h>
namespace nucray {
template <typename T> class grid : public thrust::device_vector<T> {
        size_t num_rows;
        size_t num_columns;

      public:
        grid(size_t num_rows, size_t num_columns)
            : num_rows(num_rows), num_columns(num_columns),
              thrust::device_vector<T>(num_rows * num_columns) {
        }
        size_t get_num_rows() {
              return num_rows;
        }
        size_t get_num_columns() {
              return num_columns;
        }
};
} // namespace nucray

#endif
