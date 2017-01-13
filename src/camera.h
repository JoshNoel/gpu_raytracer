#ifndef CUDA_RAYTRACER_CAMERA_H
#define CUDA_RAYTRACER_CAMERA_H

#include "cuda_runtime.h"
#include "math_helper.h"
#include <GLFW/glfw3.h>

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
		static Camera make_camera(float3 p_pos, float3 p_dir, float3 p_up, unsigned int p_width, unsigned int p_height, float p_fov = deg_to_rad(60), float m_focal_length = 1.0f);
		
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


		struct MOVE_PARAMETERS {
			static const float FORWARD_SPEED;
			static const float SIDE_SPEED;
			static const float BACK_SPEED;
		};

	};
}
#endif
