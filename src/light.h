#ifndef CUDA_RAYTRACER_LIGHT_H
#define CUDA_RAYTRACER_LIGHT_H

#include "cuda_runtime.h"

namespace cray
{
	//32 bytes
	class Light
	{
	public:
		enum LIGHT_TYPE {
			UNDEFINED = -1,
			POINT = 0,
			DIRECTIONAL = 1,
			NUM_TYPES = 2
		};

		struct PointLight {
			PointLight(float3 p_pos) { m_pos = p_pos; }
			//12 bytes
			float3 m_pos;
		};

		struct DirLight {
			DirLight(float3 p_dir) { m_dir = p_dir; }
			//12 bytes
			float3 m_dir;
		};

		Light();
		Light(const Light& p_light);
		~Light();

		//light characteristics can only be set once through these functions
		static Light make_point_light(float3 p_pos, float3 p_color, float p_intensity);
		static Light make_directional_light(float3 p_dir, float3 p_color, float p_intensity);


		__device__ const LIGHT_TYPE& get_type() const { return m_type; }
		__device__ float get_intensity() const { return m_intensity; }
		__device__ const float3& get_color() const { return m_color; }

		//don't check m_type to avoid overhead. 
		//downside is caller must verify light's type before calling either function
		__device__ const PointLight& get_point_light() const { return m_point_light; }
		__device__ const DirLight& get_dir_light() const { return m_dir_light; }

	private:
		//20 bytes
		LIGHT_TYPE m_type;
		float m_intensity;

		//TODO: implement proper color mixing. Variable is currently unused
		float3 m_color;

		union {
			PointLight m_point_light;
			DirLight m_dir_light;
		};
	};
}
#endif
