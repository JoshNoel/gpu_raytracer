#include "gl_renderer.h"
#include <fstream>
#include <string>
#include "cuda_gl_interop.h"
#include <iostream>

namespace cray
{
	gl_renderer::gl_renderer(std::string p_vert_path, std::string p_frag_path, unsigned width, unsigned height)
		: m_vert_shader_path(p_vert_path), m_frag_shader_path(p_frag_path)
	{
		glGenVertexArrays(1, &m_vertex_array_id);
		glBindVertexArray(m_vertex_array_id);

		glGenBuffers(1, &m_vertex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_rect_points), m_rect_points, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glBindVertexArray(0);

		load_shaders();

		glGenTextures(1, &m_texture);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glBindTexture(GL_TEXTURE_2D, 0);
	}


	gl_renderer::~gl_renderer() {
	}

	void gl_renderer::render() {
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(m_program_id);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glBindVertexArray(m_vertex_array_id);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		m_frame_counter++;
		if (m_frame_counter % 100 == 0) {
			m_frame_counter = 0;
			printf("\rFPS: %f", 100.0 / std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - m_last_fps_frame_time).count());
			m_last_fps_frame_time = std::chrono::high_resolution_clock::now();
		}
	}


	bool gl_renderer::load_shaders() {
		std::cout << "Loading Shaders..." << std::endl;
		std::cout << "\tOpening vertex shader." << std::endl;
		GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
		std::string vertexSource;
		std::ifstream vertexStream(m_vert_shader_path, std::ios::in);
		if(vertexStream.is_open())
		{
			std::string line = "";
			while (std::getline(vertexStream, line))
			{
				vertexSource += "\n" + line;
			}
			vertexStream.close();
		}
		else
		{
			fprintf(stderr, "Error opening vertex shader at given path: %s\n", m_vert_shader_path.c_str());
			return false;
		}
		std::cout << "\tOpening fragment shader." << std::endl;

		GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
		std::string fragSource;
		std::ifstream fragStream(m_frag_shader_path, std::ios::in);
		if (fragStream.is_open())
		{
			std::string line = "";
			while (std::getline(fragStream, line))
			{
				fragSource += "\n" + line;
			}
			fragStream.close();
		}
		else
		{
			fprintf(stderr, "Error opening fragment shader at given path: %s\n", m_frag_shader_path.c_str());
			return false;
		}

		GLint res = GL_FALSE;
		int logLength;

		std::cout << "\tCompiling vertex shader." << std::endl;
		const char* vSource = vertexSource.c_str();
		glShaderSource(vertShader, 1, &vSource, NULL);
		glCompileShader(vertShader);
		glGetShaderiv(vertShader, GL_COMPILE_STATUS, &res);
		glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLength);
		if (logLength > 0) {
			std::vector<char> errMess(logLength + 1);
			glGetShaderInfoLog(vertShader, logLength, NULL, &errMess[0]);
			printf("Vertex Shader Compile Error: %s\n", &errMess[0]);
			return false;
		}

		std::cout << "\tCompiling fragment shader." << std::endl;
		const char* fSource = fragSource.c_str();
		glShaderSource(fragShader, 1, &fSource, NULL);
		glCompileShader(fragShader);
		glGetShaderiv(fragShader, GL_COMPILE_STATUS, &res);
		glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &logLength);
		if (logLength > 0) {
			std::vector<char> errMess(logLength + 1);
			glGetShaderInfoLog(fragShader, logLength, NULL, &errMess[0]);
			printf("Fragment Shader Compile Error: %s\n", &errMess[0]);
			return false;
		}

		std::cout << "\tLinking shader program." << std::endl;
		GLuint program = glCreateProgram();
		glAttachShader(program, vertShader);
		glAttachShader(program, fragShader);

		glLinkProgram(program);
		glGetProgramiv(program, GL_LINK_STATUS, &res);
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
		if (logLength > 0) {
			std::vector<char> errMess(logLength + 1);
			glGetProgramInfoLog(program, logLength, NULL, &errMess[0]);
			printf("Shader Program Link Error: %s\n", &errMess[0]);
			return false;
		}

		glDetachShader(program, vertShader);
		glDetachShader(program, fragShader);

		glDeleteShader(vertShader);
		glDeleteShader(fragShader);

		m_program_id = program;
		return true;
	}
}

const GLfloat cray::gl_renderer::m_rect_points[30] = {
	//position				//uv-coords
	-1.0f, 1.0f, 0.0f,		0.0f, 1.0f,
	-1.0f, -1.0f, 0.0f,		0.0f, 0.0f,
	1.0f, -1.0f, 0.0f,		1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f,		0.0f, 1.0f,
	1.0f, -1.0f, 0.0f,		1.0f, 0.0f,
	1.0f, 1.0f, 0.0f,		1.0f, 1.0f
};

