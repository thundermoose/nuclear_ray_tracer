#include <blixttest/test.hh>
#include <iostream>
#include <legendre/legendre.hh>

const size_t num_points = 100;
const float tollerance = 1.0e-5;

template <typename F> void test_specific(F &&exact_formula, int l, int m) {
        std::cout << "Testing: legendre(" << l << ", " << m << ")" << std::endl;
        for (size_t i = 0; i < num_points; i++) {
                float x = 2.0 * i / (num_points - 1) - 1.0;
                float result = legendre(l, m, x);
                float exact = exact_formula(x);
                std::cout << "x = " << x << std::endl;
                std::cout << "result = " << result << std::endl;
                std::cout << "exact = " << exact << std::endl;
                std::cout << "difference = " << fabs(result - exact) << std::endl;
                assert_that(fabs(result - exact) < tollerance);
        }
}

new_test(legendre_compared_to_exact_expresions) {
        test_specific([](float x) { return 1.0f; }, 0, 0);
        test_specific([](float x) { return x; }, 1, 0);
        test_specific([](float x) { return -sqrtf(1 - x * x); }, 1, 1);
        test_specific([](float x) { return 0.5f * sqrtf(1 - x * x); }, 1, -1);
        test_specific([](float x) { return 3 * (1 - x * x); }, 2, 2);
        test_specific([](float x) { return -3 * x*sqrtf(1 - x * x); }, 2, 1);
        test_specific([](float x) { return 0.5f * (3*x * x - 1); }, 2, 0);
}
