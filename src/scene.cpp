#include "Scene.h"
#include "cuda_helper.h"
#include "gl_helper.h"

#include "pugixml.hpp"
#include "cuda_runtime.h"

#include <fstream>
#include <unordered_map>

namespace cray
{
	__device__ __constant__ Camera d_tracer_camera;

	Scene::Scene()
		: m_d_scene(nullptr)
	{
		cray::cray_key_camera = &m_camera;
	}

	Scene::Scene(std::string path) 
		: Scene() {
		loadFromFile(path);
	}


	Scene::~Scene()
	{
	}

	//kernel to 
	__global__ void init_objects() {
		
	}

	float3 xml_get_float3(pugi::xml_node node) {
		return make_float3(node.attribute("x").as_float(),
			node.attribute("y").as_float(),
			node.attribute("z").as_float());
	}

	bool Scene::loadFromFile(std::string path) {
		bool result = true;
		std::ifstream fstream;
		fstream.open(path);
		if(!fstream.is_open()) {
			fprintf(stderr, "Error opening scene file: %s\n", path.c_str());
			return false;
		}

		//allocate on host, copy to DeviceScene
		std::vector<Light> lightVector;
		std::vector<Object> objectVector;
		std::unordered_map<unsigned int, Material> materialMap;

		pugi::xml_document file;
		pugi::xml_parse_result res =  file.load_file(path.c_str());
		if(!res) {
			fprintf(stderr, "Scene file: %s, parsed with error: %s", path.c_str(), res.description());
			return false;
		}

		//load camera
		pugi::xml_node camera_node = file.child("Camera");
		unsigned int width = camera_node.attribute("width").as_uint();
		unsigned int height = camera_node.attribute("height").as_uint();
		unsigned int fov = camera_node.attribute("fov").as_uint();
		float focal_length = camera_node.attribute("focal_length").as_float();

		float3 clear_color = xml_get_float3(camera_node.child("clear_color"));
		float3 pos = xml_get_float3(camera_node.child("pos"));
		float3 point_to = xml_get_float3(camera_node.child("point_to"));
		float3 up = xml_get_float3(camera_node.child("up"));

		m_camera = Camera::make_camera(pos, point_to - pos, up, width, height, clear_color, deg_to_rad(fov), focal_length);
		copy_camera();
 
		//for each light
			//lights.push_back(make_*_light(...))
		//cudaCopy(lights.data)
		pugi::xml_node lights = file.child("Lights");
		for(auto light : lights.children("Light")) {
			float intensity = light.attribute("intensity").as_float();
			float3 color = xml_get_float3(light.child("color"));

			std::string type = light.attribute("type").as_string();
			if(type == "point") {
				float3 pos = xml_get_float3(light.child("pos"));
				lightVector.push_back(Light::make_point_light(pos, color, intensity));
			} else if (type == "dir") {
				float3 dir = xml_get_float3(light.child("dir"));
				lightVector.push_back(Light::make_directional_light(dir, color, intensity));
			} else {
				fprintf(stderr, "Error loading light from xml scene: invalid type");
				result = false;
			}
		}

		//for each material
			//materials.map(Material(r,g,b), id)
		pugi::xml_node material_node = file.child("Materials");
		for (auto material : material_node.children("Material")) {
			unsigned int id = material.attribute("id").as_uint();
			float3 color = xml_get_float3(material.child("color"));
			materialMap.try_emplace(id, Material(color));
		}
		
		//for each object
			//objectVector.push_back( loadOBJ())
		//cudaCopy(objectVector.data())
		pugi::xml_node object_node = file.child("Objects");
		for(auto obj : object_node.children()) {
			unsigned int material_id = obj.attribute("mat_id").as_uint();
			if(material_id == 0) {
				fprintf(stderr, "Error loading object from xml scene: invalid mat_id");
				result = false;
			}
			std::string obj_path = obj.attribute("path").as_string();
			if (obj_path != "") {
				objectVector.push_back(Object());
				objectVector.back().loadObj(obj_path, materialMap.at(material_id));
			}
		}

		//create DeviceScene on host for copy to device
		DeviceScene deviceScene;

		//copy light array to device
		deviceScene.m_num_lights = lightVector.size();
		Light* deviceLights = nullptr;
		CUDA_CHECK(cudaMalloc(&deviceLights, deviceScene.m_num_lights * sizeof(Light)));
		CUDA_CHECK(cudaMemcpy(deviceLights, lightVector.data(), deviceScene.m_num_lights * sizeof(Light), cudaMemcpyHostToDevice));
		deviceScene.m_lights = deviceLights;

		//copy object array to device
		deviceScene.m_num_objects = objectVector.size();
		for(auto obj : objectVector) {
			Triangle* deviceTris;
			CUDA_CHECK(cudaMalloc(&deviceTris, obj.num_tris * sizeof(Triangle)));
			CUDA_CHECK(cudaMemcpy(deviceTris, obj.tris, obj.num_tris * sizeof(Triangle), cudaMemcpyHostToDevice));
			//delete and overwrite obj->tris. The object structure can then be copied to the device
			free(obj.tris);
			obj.tris = deviceTris;
		}

		Object* deviceObjects = nullptr;
		CUDA_CHECK(cudaMalloc(&deviceObjects, deviceScene.m_num_objects * sizeof(Object)));
		CUDA_CHECK(cudaMemcpy(deviceObjects, objectVector.data(), deviceScene.m_num_objects * sizeof(Object), cudaMemcpyHostToDevice));
		deviceScene.m_objects = deviceObjects;

		CUDA_CHECK(cudaMalloc(&m_d_scene, sizeof(DeviceScene)));
		CUDA_CHECK(cudaMemcpy(m_d_scene, &deviceScene, sizeof(DeviceScene), cudaMemcpyHostToDevice));

		return result;
	}

	void Scene::copy_camera() {
		//to get rid of intellisense error
#ifdef __CUDACC__
		CUDA_CHECK(cudaMemcpyToSymbol(d_tracer_camera, &m_camera, sizeof(Camera)));
#endif
	}
}
