#ifndef CUDA_RAYTRACER_RESOURCE_MAP_H
#define CUDA_RAYTRACER_RESOURCE_MAP_H

#include <string>

//defined in this header for easy modification of resource paths on different systems
//TODO: extend in future to accept paths during project compilation and scene name as program argument
namespace cray {

	inline std::string get_shader_path(std::string shader_name) {
		return "./res/shaders/" + shader_name;
	}

	inline std::string get_object_path(std::string object_name) {
		return "./res/objects/" + object_name;
	}

	inline std::string get_scene_path(std::string scene_name) {
		return "./res/scenes/" + scene_name;
	}

}

#endif
