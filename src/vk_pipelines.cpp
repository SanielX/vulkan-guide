﻿#include <vk_pipelines.h>
#include <fstream>
#include <vk_initializers.h>

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule)
{
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	bool success = false;
	if (file.is_open()) 
	{
		size_t fileSize = (size_t)file.tellg(); // what retard names these functions?
		assert(fileSize % sizeof(uint32_t) == 0); // otherwise, not valid SPIR-V

		std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

		file.seekg(0);
		file.read((char*)buffer.data(), fileSize);

		file.close();

		VkShaderModuleCreateInfo createInfo{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .pNext = nullptr };

		createInfo.codeSize = buffer.size() * sizeof(uint32_t);
		createInfo.pCode    = buffer.data();

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) == VK_SUCCESS)
		{
			*outShaderModule = shaderModule;
			success = true;
		}
	}

	return success;
}
