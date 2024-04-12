#ifndef __VECTOR__
#define __VECTOR__
namespace nucray {
class vector {
          float x,y,z;      
          public:
                  __host__ __device__
                  vector(float x, float y, float z);

                  __host__ __device__
                  vector operator + (const vector &right_term);

                  __host__ __device__
                  vector operator - (const vector &right_term);

                  __host__ __device__
                  float operator * (const vector &right_term);

                  __host__ __device__
                  bool operator == (const vector &right_term);
};
}
#endif
