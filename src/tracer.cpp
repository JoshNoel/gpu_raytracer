#include "tracer.h"
#include "cuda_helper.h"
#include "ray.h"
#include "scene.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>

namespace cray
{


	Tracer::Tracer(Camera& p_camera, const float4& p_clear_color)
		: m_camera(p_camera), m_clear_color(p_clear_color)
	{
	}

	Tracer::~Tracer()
	{
	}








	void Tracer::register_texture(GLuint p_texture) {
		CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_image, p_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	}


	__global__ void render_kernel(const Scene::DeviceScene* p_scene) {

	}
	__global__ void render_kernel(cudaSurfaceObject_t p_surface, Sphere* p_spheres, Plane* p_planes, 
		Light* p_lights, unsigned int p_num_spheres, unsigned int p_num_planes, int p_num_lights, float4 p_clear_color) {
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		bool intersects = false;
		Ray ray = Ray::make_primary(x, y, d_tracer_camera);
		for(auto i = 0 ; i < p_num_spheres; i++) {
			if(p_spheres[i].intersects(ray))
			{
				intersects = true;
//to get rid of intellisense error
#ifdef __CUDACC__
				surf2Dwrite(p_spheres[i].calc_lighting(ray, p_lights, p_num_lights), p_surface, x * sizeof(float4), y);
#endif
			}
		}

		for (auto i = 0; i < p_num_planes; i++) {
			if (p_planes[i].intersects(ray))
			{
				intersects = true;
				//to get rid of intellisense error
#ifdef __CUDACC__
				surf2Dwrite(p_planes[i].calc_lighting(ray, p_lights, p_num_lights), p_surface, x * sizeof(float4), y);
#endif
			}
		}
		
		//if no intersection, just output clear color
		if(!intersects) {
			//to get rid of intellisense error
#ifdef __CUDACC__
			surf2Dwrite(p_clear_color, p_surface, x * sizeof(float4), y);
#endif
		}
	}

	void Tracer::render() {
		assert(m_device_pointers_initialized);
		if(m_camera.m_needs_update) {
			copy_camera();
			m_camera.m_needs_update = false;
		}
		CUDA_CHECK(cudaGraphicsMapResources(1, &m_image));
		cudaArray_t imageArray;
		CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&imageArray, m_image, 0, 0));
		cudaResourceDesc resource_desc;
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = imageArray;
		cudaSurfaceObject_t surface;

		CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resource_desc));
		//call kernel - pass surface, width, height
		//make size dynamic later
		dim3 blockDim(32, 16);
		dim3 gridDim(m_camera.m_width / 32, m_camera.m_height / 16);

//to get rid of intellisense error
#ifdef __CUDACC__
		render_kernel<<<gridDim, blockDim>>> (surface, m_d_spheres, m_d_planes, m_d_lights, m_spheres.size(), m_planes.size(), m_lights.size(), m_clear_color);
#endif
		CUDA_CHECK(cudaDestroySurfaceObject(surface));
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_image));
		CUDA_CHECK(cudaStreamSynchronize(0));
	}
}
