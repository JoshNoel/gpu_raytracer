#include "object.h"
#include <vector>
#include <fstream>
#include <iostream>

namespace cray {
	bool Object::loadObj(const std::string& path, Material& p_material) {
		material = p_material;

		std::vector<float3> vertices;
		std::vector<float> tex_coords;

		//stores indices into "vertices" and "tex_coords" that coorespond to faces of object
		std::vector<unsigned int> vertex_indices;
		std::vector<unsigned int> tex_indices;

		bool has_tex_coords;

		int extStart = path.find_last_of('.');
		if (path.substr(extStart) != ".obj")
			return false;

		std::ifstream ifs;
		ifs.open(path);
		if (!ifs.is_open())
			return false;

		std::string line = "";
		//import vertices
		while (std::getline(ifs, line)) {
			if (line[0] == 'v' && line[1] != 'n') {
				size_t sz = 1;
				float positions[3];
				for (auto i = 0; i < 3; i++) {
					positions[i] = std::stof(line.substr(sz), &sz);
				}
				vertices.push_back(make_float3(positions[0], positions[1], positions[3]));
			}

			else if (line[0] == 'f') {
				//format: vert/tex/norm
				//norm is unused in current intersection implementation
				num_tris++;
				size_t space_pos = 0;
				for (auto i = 0; i < 3; i++) {
					//finds space after current space_pos and vertex_index is next integer in the line
					vertex_indices.push_back(std::stoul(line.substr(line.find(' ', space_pos) + 1), &space_pos));

					//texture coordinate is the number after the '/' after the vertex index
				}
			}
		}

		if (vertices.size() == 0 || vertex_indices.size() == 0)
			return false;

		//create triangle array based on information
		tris = static_cast<Triangle*>(malloc(sizeof(Triangle) * num_tris));
		for (unsigned int i = 0; i < num_tris; i++) {
			//create vertices from float 
			tris[i] = Triangle::make_triangle(vertices[vertex_indices[i * 3]], vertices[vertex_indices[i * 3 + 1]], vertices[vertex_indices[i * 3 + 2]]);
		}

		return true;
	}
}