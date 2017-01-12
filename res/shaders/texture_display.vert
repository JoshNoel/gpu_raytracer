#version 450

layout(location = 0) in vec3 vertPos;
layout(location = 1) in vec2 vertUV;

out vec2 UV;

void main(){
  gl_Position = vec4(vertPos,1);
  UV = vertUV;
}