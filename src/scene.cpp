#include "Scene.h"
#include "cuda_helper.h"

#include "pugixml.hpp"
#include "cuda_runtime.h"

#include <fstream>

namespace cray
{
	__device__ __constant__ Camera d_tracer_camera;

	Scene::Scene()
		: m_d_scene(nullptr)
	{
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

		DeviceScene deviceScene;
		//allocate on host, copy to DeviceScene
		std::vector<Light> lightVector;
		std::vector<Shape*> shapeVector;

		pugi::xml_document file;
		pugi::xml_parse_result res =  file.load_file(path.c_str());
		if(!res) {
			fprintf(stderr, "Scene file: %s, parsed with error: %s", path, res.description());
			return false;
		}

		//load camera
		{
			pugi::xml_node camera_node = file.child("Camera");
			unsigned int width = camera_node.attribute("width").as_uint();
			unsigned int height = camera_node.attribute("height").as_uint();
			unsigned int fov = camera_node.attribute("fov").as_uint();
			float focal_length = camera_node.attribute("focal_length").as_float();

			float3 pos = xml_get_float3(camera_node.child("pos"));
			float3 point_to = xml_get_float3(camera_node.child("point_to"));
			float3 up = xml_get_float3(camera_node.child("up"));

			m_camera = Camera::make_camera(pos, point_to - pos, up, width, height, deg_to_rad(fov), focal_length);
		}
 
		//for each light
			//lights.push_back(make_*_light(...))
		//cudaCopy(lights.data)
		pugi::xml_node lights = file.child("Lights");
		for(auto light : lights.children("light")) {
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
		cudaMalloc(&deviceScene.m_lights, lightVector.size() * sizeof(Light));
		cudaMemcpy(deviceScene.m_lights, lightVector.data(), lightVector.size() * sizeof(Light), cudaMemcpyHostToDevice);
		
		//for each shape
			//shape_type* d_shape
			//shapesVector.push_back(d_shape)
		//cudaCopy(shapesVector.data())
		pugi::xml_node object_node = file.child("Objects");
		for(auto shape : object_node.children()) {
			std::string type = shape.name();
			unsigned int material_id = shape.attribute("mat_id").as_uint();
			if(material_id == 0) {
				fprintf(stderr, "Error loading object from xml scene: invalid mat_id");
				result = false;
			}
			if(type == "Sphere") {
				
			} else if(type == "Plane") {
				
			} else {
				fprintf(stderr, "Error loading object from xml scene: invalid type");
				result = false;
			}
		}
		return result;
	}

	__global__ init_device_scene_kernel(Shape* copy_from, Shape* copy_to, unsigned int size) {
		for(auto i = 0; i < size; i++) {
			
		}
	}

	void Scene::init_device_scene() {
		
	}


	void Scene::create_cuda_objects() {
		copy_camera();
		for(auto it = m_shapes.begin(); it !=  m_shapes.end(); ++it) {
			
		}

		CUDA_CHECK(cudaMalloc(&m_d_spheres, sizeof(Sphere) * m_spheres.size()));
		CUDA_CHECK(cudaMemcpy(m_d_spheres, m_spheres.data(), sizeof(Sphere) * m_spheres.size(), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&m_d_planes, sizeof(Plane) * m_planes.size()));
		CUDA_CHECK(cudaMemcpy(m_d_planes, m_planes.data(), sizeof(Plane) * m_planes.size(), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&m_d_lights, sizeof(Light) * m_lights.size()));
		CUDA_CHECK(cudaMemcpy(m_d_lights, m_lights.data(), sizeof(Light) * m_lights.size(), cudaMemcpyHostToDevice));
		m_device_pointers_initialized = true;
	}

	void Scene::copy_camera() {
		//to get rid of intellisense error
#ifdef __CUDACC__
		CUDA_CHECK(cudaMemcpyToSymbol(d_tracer_camera, &m_camera, sizeof(Camera)));
#endif
	}

	/*Scene::DeviceScene* Scene::get_device_scene() {
		std::vector<Shape*> shapes;
		std::vector<Light*> lights;

		//copy temporary arrays of observing pointers for copying to device
		for(auto s : m_shapes) {
			shapes.push_back(s.get());
		}
		for (auto l : m_lights) {
			lights.push_back(l.get());
		}

		Shape*
	}*/

}
