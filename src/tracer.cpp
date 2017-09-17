#include "tracer.h"
#include "cuda_helper.h"
#include "ray.h"
#include "camera.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>
#include "scene.h"

namespace cray
{
	Tracer::Tracer(Scene& p_scene)
		: m_scene(p_scene)
	{
	}

	Tracer::~Tracer()
	{
	}

	
	void Tracer::register_texture(GLuint p_texture) {
		CUDA_CHECK(cudaGraphicsGLRegisterImage(&m_image, p_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	}


	__global__ void render_kernel(cudaSurfaceObject_t p_surface, Scene::DeviceScene* p_scene) {
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		bool intersects = false;
        Ray ray = Ray::make_primary(x, y, p_scene->m_p_camera);
        for (auto i = 0; i < p_scene->m_num_objects; i++) {
            if (p_scene->m_objects[i].intersects(ray))
            {
                intersects = true;
            }
        }

        //if no intersection, just output clear color
        if (!intersects) {
            //to get rid of intellisense error
#ifdef __CUDACC__
            surf2Dwrite(p_scene->m_p_camera->get_clear_color(), p_surface, x * sizeof(float4), y);
#endif
        }
        else {
            //to get rid of intellisense error
#ifdef __CUDACC__
            surf2Dwrite(ray.get_hit_object()->calc_lighting(ray, p_scene->m_lights, p_scene->m_num_lights), p_surface, x * sizeof(float4), y);
#endif
        }
	}

	void Tracer::render() {
		assert(m_scene.get_device_scene());
		m_scene.update_camera();

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
		dim3 gridDim(m_scene.m_camera.get_width() / 32, m_scene.m_camera.get_height() / 16);

//to get rid of intellisense error
#ifdef __CUDACC__
		render_kernel<<<gridDim, blockDim>>> (surface, m_scene.get_device_scene());
#endif
		CUDA_CHECK(cudaDestroySurfaceObject(surface));
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_image));
		CUDA_CHECK(cudaStreamSynchronize(0));
	}
}
