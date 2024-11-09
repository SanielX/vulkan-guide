#pragma once 
#include <vk_types.h>

namespace vkutil 
{
	// Loads SPIR-V shader
	bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);
};