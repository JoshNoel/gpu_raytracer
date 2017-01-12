#include "gl_helper.h"
#include <iostream>
#include <stdlib.h>

namespace cray
{
	void CALLBACK opengl_error_callback(GLenum source​, GLenum type, GLuint id​, GLenum severity, GLsizei length​, const GLchar * p_message, const void * userParam​)
	{
		std::cout << "------------------------OPENGL ERROR--------------------------" << std::endl;
		std::cout << "Message: " << p_message << std::endl;
		std::cout << "Type: ";
		switch (type)
		{
		case GL_DEBUG_TYPE_ERROR:
			std::cout << "ERROR";
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			std::cout << "DEPRECATED";
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			std::cout << "UNDEFINED";
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			std::cout << "PORTABILITY";
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			std::cout << "PERFORMANCE";
			break;
		case GL_DEBUG_TYPE_MARKER:
			std::cout << "MARKER";
			break;
		}
		std::cout << std::endl;
		std::cout << "Severity: ";
		bool willAbort = false;
		switch (severity)
		{
		case GL_DEBUG_SEVERITY_HIGH:
			std::cout << "HIGH";
			willAbort = true;
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			std::cout << "MEDIUM";
			break;
		case GL_DEBUG_SEVERITY_LOW:
			std::cout << "LOW";
			break;
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			std::cout << "NOTIFICATION";
			break;
		}
		std::cout << std::endl;
#ifdef BREAK_ON_GL_ERROR
		if (willAbort) {
			std::cout << "Aborting..." << std::endl;
			abort();
		}
#endif;
		std::cout << "--------------------------------------------------------------" << std::endl << std::endl;

	}
}