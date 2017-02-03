#ifndef CUDA_RAYTRACER_GL_HELPER_H
#define CUDA_RAYTRACER_GL_HELPER_H

#include <Windows.h>
#include <iostream>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "camera.h"

namespace cray
{
	//opengl error callback
	void CALLBACK opengl_error_callback(GLenum p_source, GLenum p_type, GLuint p_id, GLenum p_severity, GLsizei p_length,
		const GLchar* const p_message, const void* const p_userParam);

	void key_handler(GLFWwindow* p_window);
	extern Camera* cray_key_camera;
}

#endif
