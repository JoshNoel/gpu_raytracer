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

		Light();
		Light(const Light& p_light);
		~Light();

		//20 bytes
		LIGHT_TYPE m_type;
		float m_intensity;

		//TODO: implement proper color mixing. Variable is currently unused
		float3 m_color;

		union {
			PointLight m_point_light;
		};

		static Light make_point_light(float3 p_pos, float3 p_color, float p_intensity);
	};
}
#endif
