#ifndef CUDA_RAYTRACER_MATH_HELPER_H
#define CUDA_RAYTRACER_MATH_HELPER_H

#include "cuda_runtime.h"
#include <cmath>

namespace cray
{
	const float _PI_ = 3.1415f;

    struct quat {
        union {
            struct {
                float w, x, y, z;
            };
            struct {
                float r;
                float3 v;
            };
        };
    };

    ///// UTIL FUNCS //////
    static inline __host__ __device__ char sign(float a) {
        return a >= 0 ? 1 : -1;
    }

    static inline __host__ float deg_to_rad(float a) {
        return a * _PI_ / 180.0f;
    }
    ///////////////////////

	//all inputs are treated as float3, regardless of dimension
	//allows one to load float4 (less instructions), but still use 3D operators
	
    /////// BASIC VEC OPS (FLOAT2) //////////
	static inline __host__ __device__ float2 operator*(const float2& a, const float& b) {
		return make_float2(a.x * b, a.y * b);
	}

	static inline __host__ __device__ float2 operator/(const float2& a, const float& b) {
		return make_float2(a.x / b, a.y / b);
	}
    
    static inline __host__ __device__ float2 operator*(const float& a, const float2& b) {
        return b*a;
    }

    static inline __host__ __device__ float2 operator/(const float& a, const float2& b) {
        return b/a;
    }

	static inline __host__ __device__ float2 operator+(const float2& a, const float2& b) {
		return make_float2(a.x + b.x, a.y + b.y);
	}

	static inline __host__ __device__ float2 operator-(const float2& a, const float2& b) {
		return make_float2(a.x - b.x, a.y - b.y);
	}

    /////// BASIC VEC OPS (FLOAT3) //////////
	static inline __host__ __device__ float3 operator*(const float3& a, const float& b) {
		return make_float3(a.x * b, a.y * b, a.z * b);
	}

	static inline __host__ __device__ float3 operator/(const float3& a, const float& b) {
		return make_float3(a.x / b, a.y / b, a.z / b);
	}

    static inline __host__ __device__ float3 operator*(const float& a, const float3& b) {
        return b*a;
    }

    static inline __host__ __device__ float3 operator/(const float& a, const float3& b) {
        return b / a;
    }


	static inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

    static inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    /////// BASIC VEC OPS (FLOAT4) //////////
    static inline __host__ __device__ float4 operator*(const float4& a, const float& b) {
        return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
    }

    static inline __host__ __device__ float4 operator/(const float4& a, const float& b) {
        return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
    }

    static inline __host__ __device__ float4 operator*(const float& a, const float4& b) {
        return b*a;
    }

    static inline __host__ __device__ float4 operator/(const float& a, const float4& b) {
        return b / a;
    }

    static inline __host__ __device__ float4 operator+(const float4& a, const float4& b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.z + b.z);
    }

    static inline __host__ __device__ float4 operator-(const float4& a, const float4& b) {
        return make_float4(a.x - b.x, a.y - b.y, a.z + b.z, a.w - b.w);
    }

    static inline __host__ __device__ float4 operator*(const float4& a, const float4& b) {
        return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.z * b.z);
    }

    static inline __host__ __device__ float4 operator/(const float4& a, const float4& b) {
        return make_float4(a.x / b.x, a.y / b.y, a.z + b.z, a.w / b.w);
    }

    // Allow ops between float3 and float4. Allows us to store tri coords in float4 for faster operations, 
    // but also to interact with them as if they were float3;
    static inline __host__ __device__ float3 operator-(const float3& a, const float4& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

	/////////////////////////////////

	///// DOT PRODUCT //////
	static inline __host__ __device__  float dot(const float3& a, const float3& b) {
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}
	static inline __host__ __device__  float dot(const float4& a, const float4& b) {
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}
	static inline __host__ __device__  float dot(const float3& a, const float4& b) {
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}
	static inline __host__ __device__  float dot(const float4& a, const float3& b) {
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}
	///////////////////////

	//// CROSS  PRODUCT ////
	static inline __host__ __device__  float3 cross(const float3& a, const float3& b) {
		return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
	}
	static inline __host__ __device__  float3 cross(const float4& a, const float4& b) {
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}
	static inline __host__ __device__  float3 cross(const float3& a, const float4& b) {
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}
	static inline __host__ __device__  float3 cross(const float4& a, const float3& b) {
		return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}
	/////////////////////

    ////// VEC OPS (Generic) /////
    static inline __host__ __device__ float mag(const float3& a) {
        //to get rid of intellisense error
#ifdef __CUDACC__
        return sqrtf(dot(a, a));
#else
        return sqrtf(dot(a, a));
#endif
    }

    static inline __host__ __device__ float3 norm(const float3& a) {
        return a / mag(a);
    }

    ////// QUAT OPS ///////
    static inline __host__ __device__ quat make_quat(const float3& a) {
        quat res;
        res.r = 0;
        res.v = a;
        return res;
    }

    static inline __host__ __device__ quat operator*(const quat& a, const quat& b) {
        //quaternion muliplication: source(https://en.wikipedia.org/wiki/Quaternion)
        quat res;
        res.r = a.r*b.r - dot(a.v, b.v);
        res.v = a.r*b.v + b.r*a.v + cross(a.v, b.v);
        return res;
    }

    static inline __host__ __device__ float3 operator*(const quat& a, const float3& b) {
        quat b_quat = make_quat(b);
        quat res = a * b_quat;
        return res.v;
    }

    static inline __host__ __device__ float3 operator*(const float3& a, const quat& b) {
        quat a_quat = make_quat(a);
        quat res = a_quat * b;
        return res.v;
    }

    static inline __host__ __device__ quat operator/(const quat& a, float x) {
        quat res = a;
        res.r /= x;
        res.v = res.v / x;
        return res;
    }

    

    static inline __host__ __device__ float norm(const quat& q) {
        return pow(q.w, 2) + pow(q.x, 2) + pow(q.y, 2) + pow(q.z, 2);
    }

    static inline __host__ __device__ quat conj(const quat& q) {
        quat res = q;
        res.x = -res.x;
        res.y = -res.y;
        res.z = -res.z;
        return res;
    }

    static inline __host__ __device__ quat inv(const quat& q) {
        return conj(q) / norm(q);
    }
    
    static inline __host__ __device__ float3 rot(const float3& v, const quat& q) {
        //quat v_quat = make_quat(v);
        //return quat{ q*v_quat*inv(q) }.v;
        return 2.0f * dot(q.v, v) * q.v + (q.r*q.r - dot(q.v, v)) * v + 2.0f * q.r * cross(q.v, v);
    }

    static inline __host__ __device__ quat axis_angle(float3 axis, float angle) {
        quat res;
        res.r = cos(angle / 2.0f);
        float3 v = axis*sin(angle / 2.0f);
        res.v = v;
        return res;
    }
    ///////////////
	
}
#endif