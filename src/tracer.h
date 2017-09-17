#ifndef CUDA_RAYTRACER_TRACER_H
#define CUDA_RAYTRACER_TRACER_H

#include "GL/glew.h"
#include "cuda_gl_interop.h"

#include "scene.h"

namespace cray
{
	extern 	__device__ __constant__ Camera d_tracer_camera;

	class Tracer
	{
	public:
		Tracer(Scene& p_scene);
		~Tracer();

		void register_texture(GLuint p_texture);
		void render();

	private:
		cudaGraphicsResource* m_image = nullptr;
		Scene& m_scene;
	};
}
#endif