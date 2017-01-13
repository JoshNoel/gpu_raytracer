#include "camera.h"
#include "cuda_runtime.h"
#include <math.h>

namespace cray
{
	Camera Camera::make_camera(float3 p_pos, float3 p_dir, float3 p_up, unsigned p_width, unsigned p_height, float p_fov, float p_focal_length) {
		Camera cam;
		cam.m_position = p_pos;
		cam.m_dir = norm(p_dir);
		cam.m_up = norm(p_up);
		cam.m_right = norm(cross(cam.m_dir, cam.m_up));
		cam.m_width = p_width;
		cam.m_height = p_height;
		cam.m_fov = p_fov;
		cam.m_focal_length = p_focal_length;
		cam.m_ar = float(p_width) / float(p_height);
		cam.m_world_width = 2.0f * tanf(p_fov / 2.0f) * p_focal_length;
		cam.m_world_height = cam.m_world_width / cam.m_ar;
		return cam;
	}

	Camera& Camera::operator=(const Camera& p_camera) {
		this->m_position = p_camera.m_position;
		this->m_dir = norm(p_camera.m_dir);
		this->m_up = norm(p_camera.m_up);
		this->m_right = norm(p_camera.m_right);
		this->m_width = p_camera.m_width;
		this->m_height = p_camera.m_height;
		this->m_focal_length = p_camera.m_focal_length;
		this->m_fov = p_camera.m_fov;
		this->m_ar = p_camera.m_ar;
		this->m_world_width = p_camera.m_world_width;
		this->m_world_height = p_camera.m_world_height;
		return *this;
	}


	const float Camera::MOVE_PARAMETERS::FORWARD_SPEED = 0.1f;
	const float Camera::MOVE_PARAMETERS::SIDE_SPEED = 0.1f;
	const float Camera::MOVE_PARAMETERS::BACK_SPEED = 0.1f;

}