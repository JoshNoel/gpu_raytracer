#ifndef CUDA_RAYTRACER_CAMERA_H
#define CUDA_RAYTRACER_CAMERA_H

#include "cuda_runtime.h"
#include "math_helper.h"

namespace cray
{
	class Camera
	{
	public:
		//can't define constructors if camera is allocated in constant memory
			//no dynamic allocation
			//w/o default ctor compiler complains
			//with "no dynamic allocation for __constant__ variables" 
		Camera& operator=(const Camera&);
		static Camera make_camera(float3 p_pos, float3 p_dir, float3 p_up, unsigned int p_width, unsigned int p_height, float3 clear_color, float p_fov = deg_to_rad(60), 
			float m_focal_length = 1.0f);
		

		struct MOVE_PARAMETERS {
			static const float FORWARD_SPEED;
			static const float SIDE_SPEED;
			static const float BACK_SPEED;
		};

		__device__ __host__ const float3& get_position() const { return m_position; }
		void set_position(const float3& p_pos) { m_position = p_pos; }

		__device__ __host__ const float3& get_dir() const { return m_dir; }
        void set_dir(const float3& p_dir) { m_dir = p_dir; }

		__device__ __host__ const float3& get_up() const { return m_up; }
        void set_up(const float3& p_up) { m_up = p_up; }
		 
		__device__ __host__ const float3& get_right() const { return m_right; }
        void set_right(const float3& p_right) { m_right = p_right; }

		__device__	float get_focal_length() const { return m_focal_length; }

		__device__ __host__ unsigned int get_width() const { return m_width; }

		__device__ __host__ unsigned int get_height() const { return m_height; }

		__device__ float get_world_width() const { return m_world_width; }

		__device__ float get_world_height() const { return m_world_height; }

		__device__ float get_ar() const { return m_ar; }

		__device__ float get_fov() const { return m_fov; }

		bool get_needs_update() const { return m_needs_update; }
		void set_needs_update(bool p_needs_update) { m_needs_update = p_needs_update; }

		__device__ const float4& get_clear_color() const { return make_float4(m_clear_color.x, m_clear_color.y, m_clear_color.z, 0); }

	private:
		float3 m_position;
		float3 m_dir;
		float3 m_up;
		float3 m_right;
		float m_focal_length;
		unsigned int m_width, m_height;
		float m_world_width, m_world_height;
		float m_ar;

		//horizontal FOV
		float m_fov;

		bool m_needs_update;

		float3 m_clear_color;


	};
}
#endif
