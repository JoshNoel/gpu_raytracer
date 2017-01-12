#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "gl_renderer.h"
#include "gl_helper.h"
#include "camera.h"
#include "sphere.h"
#include "tracer.h"
#include "material.h"

#include <iostream>

int main() {
	//for now dimensions must be multiples of 32
	const unsigned int WIDTH = 800;
	const unsigned int HEIGHT = 800;

	glewExperimental = GL_TRUE;

	if (!glfwInit()) {
		fprintf(stderr, "Error initializing GLFW at: %s - %u \n ", __FILE__, __LINE__);
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 1);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef _DEBUG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Output", nullptr, nullptr);
	if(!window) {
		fprintf(stderr, "Error creating GLFW window at: %s - %u \n", __FILE__, __LINE__);
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	GLenum glewErr = glewInit();
	if (glewErr != GLEW_OK) {
		fprintf(stderr, "Error initializing GLEW at: %s - %u \n Error: %s \n", __FILE__, __LINE__, glewGetErrorString(glewErr));
		return -1;
	}

	const GLubyte* renderer_name = glGetString(GL_RENDERER);
	const GLubyte* version = glGetString(GL_VERSION);
	printf("Renderer: %s\nOpenGL Version: %s\n", renderer_name, version);


#ifdef _DEBUG
	if(glDebugMessageCallback)
	{
		std::cout << "Registering OpenGL Debug Callback..." << std::endl;
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(cray::opengl_error_callback, nullptr);
		GLuint ids = 0;
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, &ids, true);
	} 
	else
	{
		std::cout << "Could not register debug callback!" << std::endl;
	}
#endif

	cray::gl_renderer renderer("./res/shaders/texture_display.vert", "./res/shaders/texture_display.frag", WIDTH, HEIGHT);
	
	cray::Material material(make_float3(1.0f, 0.0f, 0.0f));
	cray::Sphere sphere(1.0f, make_float3(0.0f, 0.0f, -5.0f), material);
	cray::Light point_light;
	point_light = cray::Light::make_point_light(make_float3(0.5f, 1.0f, -1.0f), make_float3(1.0f, 1.0f, 1.0f), 1.0f);

	cray::Camera camera = cray::Camera::make_camera(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), WIDTH, HEIGHT);
	cray::Tracer tracer(camera, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
	tracer.add_sphere(sphere);
	tracer.add_light(point_light);
	tracer.create_cuda_objects();
	tracer.register_texture(renderer.m_texture);

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	cray::cray_key_camera = &camera;
	glfwSetKeyCallback(window, cray::key_callback);
	std::chrono::high_resolution_clock::time_point last_poll_time = std::chrono::high_resolution_clock::now();
	while(!glfwWindowShouldClose(window))
	{
		//trace scene -> texture
		//render full screen rectangle with texture
		tracer.render();
		renderer.render();
		glfwSwapBuffers(window);
		if (std::chrono::duration_cast<std::chrono::duration<double>>( std::chrono::high_resolution_clock::now() - last_poll_time).count() > (1.0 / 30.0))
		{
			glfwPollEvents();
			last_poll_time = std::chrono::high_resolution_clock::now();
		}
	}

	glfwTerminate();
	return 0;
}
