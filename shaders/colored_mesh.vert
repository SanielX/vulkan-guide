#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

struct Vertex 
{
	vec3  position;
	float uv_x;
	
	vec3  normal;
	float uv_y;

	vec4 color;
};

// GLSL is actually retarded I swear
layout(buffer_reference, std430) readonly buffer VertexBuffer
{
	Vertex vertices[];
};

layout(push_constant) uniform constants 
{
	mat4 		 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main() 
{
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

    gl_Position = PushConstants.render_matrix * vec4(v.position, 1.0);
    outColor = v.color.xyz;
    outUV.xy = vec2(v.uv_x, v.uv_y);
}