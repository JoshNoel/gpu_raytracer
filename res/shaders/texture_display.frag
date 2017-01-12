#version 450

in vec2 UV;
out vec4 color;

uniform sampler2D sampler;
void main() {
  //color = vec4(1.0,0.5,0.2,1.0);
  color = texture(sampler, UV);
}