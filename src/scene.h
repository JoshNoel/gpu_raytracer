#ifndef CUDA_RAYTRACER_SCENE_H
#define CUDA_RAYTRACER_SCENE_H

#include <string>
#include <vector>
#include <memory>

#include "object.h"
#include "light.h"
#include "camera.h"

namespace cray
{
	class Scene
	{
	public:
		struct DeviceScene {
            Camera* m_p_camera;
			Object* m_objects;
			Light* m_lights;
			unsigned int m_num_objects;
			unsigned int m_num_lights;
		};

		Scene();
		Scene(std::string path);
		~Scene();

		//loads objects from scene description file to device
		//TODO: multithread
		bool loadFromFile(std::string path);
		DeviceScene* get_device_scene() { return m_d_scene; }
		void copy_camera();
		void update_camera() {
			if(m_camera.get_needs_update()) {
				copy_camera();
				m_camera.set_needs_update(false);
			}
		}

		Camera m_camera;
        Camera* m_d_camera;

	private:
		DeviceScene* m_d_scene;
	};
}
#endif