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

		void add_sphere(const Sphere& p_sphere);
		void add_plane(const Plane& p_plane);
		void add_light(const Light& p_light);
		void register_texture(GLuint p_texture);
		void create_cuda_objects();
		void render();

	private:
		void copy_camera();

		cudaGraphicsResource* m_image = nullptr;

		Camera& m_camera;
		std::vector<Sphere> m_spheres;
		std::vector<Plane> m_planes;
		std::vector<Light> m_lights;


		Sphere* m_d_spheres = nullptr;
		Plane* m_d_planes = nullptr;
		Light* m_d_lights = nullptr;

		//to ensure device pointers are initialized before rendering
		bool m_device_pointers_initialized = false;

		float4 m_clear_color;
	};
}
#endif