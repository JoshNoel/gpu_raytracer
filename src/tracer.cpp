#include "tracer.h"
#include "cuda_helper.h"
#include "ray.h"
#include "camera.h"
#include "sphere.h"
#include "plane.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>

namespace cray
{

	__device__ __constant__ Camera d_tracer_camera;

	Tracer::Tracer(Camera& p_camera, const float4& p_clear_color)
		: m_camera(p_camera), m_clear_color(p_clear_color)
	{
	}

	Tracer::~Tracer()
	{
		cudaFree(m_d_spheres);
	}

	void Tracer::add_sphere(const Sphere& p_sphere) {
		m_spheres.push_back(p_sphere);
	}

	void Tracer::add_plane(const Plane& p_plane) {
		m_planes.push_back(p_plane);
	}


	void Tracer::add_light(const Light& p_light) {
		m_lights.push_back(p_light);
	}


	void Tracer::create_cuda_objects() {
		copy_camera();
		CUDA_CHECK(cudaMalloc(&m_d_spheres, sizeof(Sphere) * m_spheres.size()));
		CUDA_CHECK(cudaMemcpy(m_d_spheres, m_spheres.data(), sizeof(Sphere) * m_spheres.size(), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&m_d_planes, sizeof(Plane) * m_planes.size()));
		CUDA_CHECK(cudaMemcpy(m_d_planes, m_planes.data(), sizeof(Plane) * m_planes.size(), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&m_d_lights, sizeof(Light) * m_lights.size()));
		CUDA_CHECK(cudaMemcpy(m_d_lights, m_lights.data(), sizeof(Light) * m_lights.size(), cudaMemcpyHostToDevice));
		m_device_pointers_initialized = true;
	}

	void Tracer::copy_camera() {
		//to get rid of intellisense error
#ifdef __CUDACC__
		CUDA_CHECK(cudaMemcpyToSymbol(d_tracer_camera, &m_camera, sizeof(Camera)));
#endif
	}



	void Tracer::register_texture(GLuint p_texture) {
		CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_image, p_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
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
