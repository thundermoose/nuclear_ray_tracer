#include <blixttest/test.hh>
#include <cmath>
#include <thrust/detail/vector_base.h>
#include <vector/vector.hh>




new_test(adding_vectors) {
        assert_that(nucray::vector(3, 4, 5) + nucray::vector(3, 2, 1) ==
                    nucray::vector(6, 6, 6));
}

new_test(subtracting_vectors) {
        assert_that(nucray::vector(3, 4, 5) - nucray::vector(1, 2, 3) ==
                    nucray::vector(2, 2, 2));
}

new_test(scalarproduct) {
        assert_that(fabs(nucray::vector(1, 1, 1) * nucray::vector(1, 1, 1) -
                         3) < tollerance);
}

new_test(scale_vector) {
        assert_that(nucray::vector(1, 2, 3) * 2 == nucray::vector(2, 4, 6));
}

new_test(crossproduct) {
        assert_that(nucray::vector(1, 0, 0).cros(nucray::vector(0, 1, 0)) ==
                    nucray::vector(0, 0, 1));
        assert_that(nucray::vector(0, 1, 0).cros(nucray::vector(0, 0, 1)) ==
                    nucray::vector(1, 0, 0));
        assert_that(nucray::vector(0, 0, 1).cros(nucray::vector(1, 0, 0)) ==
                    nucray::vector(0, 1, 0));
        assert_that(nucray::vector(0, 1, 0).cros(nucray::vector(1, 0, 0)) ==
                    nucray::vector(0, 0, -1));
        assert_that(nucray::vector(0, 0, 1).cros(nucray::vector(0, 1, 0)) ==
                    nucray::vector(-1, 0, 0));
        assert_that(nucray::vector(1, 0, 0).cros(nucray::vector(0, 0, 1)) ==
                    nucray::vector(0, -1, 0));
}
