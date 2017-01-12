#include "tracer.h"
#include "cuda_helper.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ray.h"
#include "camera.h"
#include "sphere.h"

namespace cray
{

	__device__ __constant__ Camera d_tracer_camera;

	Tracer::Tracer(const Camera& p_camera)
		: m_camera(p_camera)
	{
	}

	Tracer::~Tracer()
	{
		cudaFree(m_d_spheres);
	}

	void Tracer::add_sphere(const Sphere& p_sphere) {
		m_spheres.push_back(p_sphere);
	}


	void Tracer::create_cuda_objects() {
		CUDA_CHECK(cudaMemcpyToSymbol(d_tracer_camera, &m_camera, sizeof(Camera)));
		CUDA_CHECK(cudaMalloc(&m_d_spheres, sizeof(Sphere) * m_spheres.size()));
		CUDA_CHECK(cudaMemcpy(m_d_spheres, m_spheres.data(), sizeof(Sphere) * m_spheres.size(), cudaMemcpyHostToDevice));
	}


	void Tracer::register_texture(GLuint p_texture) {
		CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_image, p_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	}


	__global__ void render_kernel(cudaSurfaceObject_t p_surface, Sphere* p_spheres, unsigned int p_num_spheres) {
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		Ray ray = Ray::make_primary(x, y, d_tracer_camera);
		for(auto i = 0 ; i < p_num_spheres; i++) {
			if(p_spheres[i].intersects(ray))
			{
//to get rid of intellisense error
#ifdef __CUDACC__
				surf2Dwrite(p_spheres[i].m_color, p_surface, x * sizeof(float4), y);
#endif
			}
		}
	}

	void Tracer::render() {
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

		render_kernel<<<gridDim, blockDim>>> (surface, m_d_spheres, m_spheres.size());
		CUDA_CHECK(cudaDestroySurfaceObject(surface));
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_image));
		CUDA_CHECK(cudaStreamSynchronize(0));
	}
}
