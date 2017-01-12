#ifndef CUDA_RAYTRACER_TRACER_H
#define CUDA_RAYTRACER_TRACER_H

#include "GL/glew.h"
#include "cuda_gl_interop.h"
#include "camera.h"
#include "sphere.h"
#include <vector>

namespace cray
{
	class Tracer
	{
	public:
		Tracer(const Camera& p_camera);
		~Tracer();

		void add_sphere(const Sphere& p_sphere);
		void register_texture(GLuint texture);
		void create_cuda_objects();
		void render();

	private:
		cudaGraphicsResource* m_image;

		Camera m_camera;
		std::vector<Sphere> m_spheres;

		Sphere* m_d_spheres;
	};
}
#endif