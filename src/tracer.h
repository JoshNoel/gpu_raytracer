#ifndef CUDA_RAYTRACER_TRACER_H
#define CUDA_RAYTRACER_TRACER_H

#include "GL/glew.h"
#include "cuda_gl_interop.h"

#include "camera.h"
#include "sphere.h"
#include "Light.h"
#include "plane.h"

#include <vector>

namespace cray
{
	class Tracer
	{
	public:
		Tracer(Camera& p_camera, const float4& p_clear_color);
		~Tracer();

		void register_texture(GLuint p_texture);
		void render();

	private:
		void copy_camera();

		cudaGraphicsResource* m_image = nullptr;		

		float4 m_clear_color;
	};
}
#endif