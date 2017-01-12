#ifndef CUDA_RAYTRACER_GL_RENDERER_H
#define CUDA_RAYTRACER_GL_RENDERER_H

#include "GL/glew.h"
#include <vector>

namespace cray
{
	class gl_renderer
	{
	public:
		gl_renderer(std::string p_vert_path, std::string p_frag_path, unsigned width, unsigned height);
		~gl_renderer();

		void render() const;

		GLuint m_texture;

	private:
		bool load_shaders();

		GLuint m_vertex_array_id;
		GLuint m_vertex_buffer;

		static const GLfloat m_rect_points[30];
		std::string m_vert_shader_path;
		std::string m_frag_shader_path;

		GLuint m_program_id;
	};
}
#endif