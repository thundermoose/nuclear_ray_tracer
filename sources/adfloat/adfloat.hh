#ifndef __ADFLOAT__
#define __ADFLOAT__

namespace nucray {
struct adfloat {
        float value, derivative;       
        __host__ __device__ adfloat operator + (const adfloat other) {
                adfloat result = {
                        .value = value + other.value,
                        .derivative = derivative + other.derivative
                };
                return result;
        }
        __host__ __device__ adfloat operator - (const adfloat other) {
                adfloat result = {
                        .value = value - other.value,
                        .derivative = derivative - other.derivative
                };
                return result;
        }
        __host__ __device__ adfloat operator * (const adfloat other) {
                adfloat result = {
                        .value = value*other.value,
                        .derivative = value*other.derivative + derivative*other.value
                };
                return result;
        }
        __host__ __device__ adfloat operator / (const adfloat other) {
                adfloat result = {
                        .value = value/other.value,
                        .derivative = (-value*other.derivative + derivative*other.value)/(other.value*other.value)
                };
                return result;
        }
        __host__ __device__ adfloat exp() {
                adfloat result = {
                        .value = expf(value),
                        .derivative = derivative*expf(value)
                };
                return result;
        }
        __host__ __device__ adfloat sin() {
                adfloat result = {
                        .value = sinf(value),
                        .derivative = derivative*cos(value)      
                };
                return result;
        }
        __host__ __device__ adfloat cos() {
                adfloat result = {
                        .value = cosf(value),
                        .derivative = -derivative*sin(value)
                };
                return result;
        }
        __host__ __device__ adfloat sqrt() {
                adfloat result = {
                        .value = sqrtf(value),
                        .derivative = derivative/sqrtf(value)      
                };
                return result;
        }
        __host__ __device__ adfloat powi() {
                
        }

};
}
#endif
