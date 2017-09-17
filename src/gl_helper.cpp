#include "gl_helper.h"
#include <iostream>
#include <stdlib.h>
#include "camera.h"

namespace cray
{
	Camera* cray_key_camera;
    double m_prev_mouse_x = 0.0;
    double m_prev_mouse_y = 0.0;

	void CALLBACK opengl_error_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* const p_message, const void* const userParam) {
		std::cout << "------------------------OPENGL ERROR--------------------------" << std::endl;
		std::cout << "Message: " << *p_message << std::endl;
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
	
	void key_handler(GLFWwindow* p_window) {
        //////App Control
        int control = glfwGetKey(p_window, GLFW_KEY_ESCAPE);
        if (control == GLFW_PRESS || control == GLFW_REPEAT)
        {
            glfwSetInputMode(p_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        //////Movement
		int action = glfwGetKey(p_window, GLFW_KEY_W);
		if(action == GLFW_PRESS || action == GLFW_REPEAT)
		{
            //printf("%f\n", cray_key_camera->get_dir() * Camera::MOVE_PARAMETERS::BACK_SPEED);
			cray_key_camera->set_position(cray_key_camera->get_position() + cray_key_camera->get_dir() * Camera::MOVE_PARAMETERS::FORWARD_SPEED);
			cray_key_camera->set_needs_update(true);
		}

		action = glfwGetKey(p_window, GLFW_KEY_S);
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
            //printf("%f\n", -1 * cray_key_camera->get_dir() * Camera::MOVE_PARAMETERS::BACK_SPEED);
			cray_key_camera->set_position(cray_key_camera->get_position() - cray_key_camera->get_dir() * Camera::MOVE_PARAMETERS::BACK_SPEED);
			cray_key_camera->set_needs_update(true);
		}

		action = glfwGetKey(p_window, GLFW_KEY_D);
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			cray_key_camera->set_position(cray_key_camera->get_position() + cray_key_camera->get_right() * Camera::MOVE_PARAMETERS::SIDE_SPEED);
			cray_key_camera->set_needs_update(true);
		}

		action = glfwGetKey(p_window, GLFW_KEY_A);
		if (action == GLFW_PRESS || action == GLFW_REPEAT)
		{
			cray_key_camera->set_position(cray_key_camera->get_position() - cray_key_camera->get_right() * Camera::MOVE_PARAMETERS::SIDE_SPEED);
			cray_key_camera->set_needs_update(true);
		}
	}

    void mouse_handler(GLFWwindow* window) {
        ////////////BUTTONS////////////
        int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        if (state == GLFW_PRESS)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        float2 delta;
        delta.x = xpos - m_prev_mouse_x;
        delta.y = ypos - m_prev_mouse_y;

        if (delta.y > 0 || delta.x > 0) {
            delta = -1 * delta * cray::MOUSE_SENS;

            quat rot_right = axis_angle(cray_key_camera->get_right(), delta.y);
            quat rot_up = axis_angle(cray_key_camera->get_up(), delta.x);
            quat rot_prod = rot_right * rot_up;
            //printf("||rot_prod|| = %f\n", norm(rot_prod));

            //printf("delta_y: %f\n", delta.y);
            cray_key_camera->set_dir(rot(cray_key_camera->get_dir(), rot_prod));
            //printf("dir: (%f, %f, %f)\n", cray_key_camera->get_dir().x, cray_key_camera->get_dir().y, cray_key_camera->get_dir().z);

            cray_key_camera->set_right(rot(cray_key_camera->get_right(), rot_prod));
            //printf("right: (%f, %f, %f)\n", cray_key_camera->get_right().x, cray_key_camera->get_right().y, cray_key_camera->get_right().z);

            cray_key_camera->set_up(rot(cray_key_camera->get_up(), rot_prod));

            m_prev_mouse_x = xpos;
            m_prev_mouse_y = ypos;
            cray_key_camera->set_needs_update(true);
        }
        //printf("dir: (%f, %f, %f)\n", cray_key_camera->get_dir().x, cray_key_camera->get_dir().y, cray_key_camera->get_dir().z);
    }
}

