#ifndef CUDA_RAYTRACER_GL_HELPER_H
#define CUDA_RAYTRACER_GL_HELPER_H

#include "GL/glew.h"
#include <Windows.h>


namespace cray
{
//opengl error callback
	void CALLBACK opengl_error_callback(GLenum source​, GLenum type​, GLuint id​, GLenum severity​, GLsizei length​, const GLchar* message​, const void* userParam​);
}

#endif
