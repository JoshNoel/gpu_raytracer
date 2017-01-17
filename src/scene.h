#ifndef CUDA_RAYTRACER_SCENE_H
#define CUDA_RAYTRACER_SCENE_H

#include <string>
#include <vector>
#include <memory>

#include "shape.h"
#include "light.h"
#include "camera.h"

namespace cray
{
	extern 	__device__ __constant__ Camera d_tracer_camera;

	class Scene
	{
	public:
		struct DeviceScene {
			Shape** m_shapes;
			Light* m_lights;
		};

		Scene();
		Scene(std::string path);
		~Scene();

		//loads objects from scene description file to device
		//TODO: multithread
		bool loadFromFile(std::string path);
		DeviceScene* get_device_scene();


	private:
		void copy_camera();
		void init_device_scene();


		Camera m_camera;

		//to ensure device pointers are initialized before rendering
		bool m_device_pointers_initialized = false;

		DeviceScene* m_d_scene;
	};
}
#endif