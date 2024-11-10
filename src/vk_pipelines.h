#pragma once 
#include <vk_types.h>

namespace vkutil 
{
	struct GraphicsPipelineBuilder
	{
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly;
		VkPipelineRasterizationStateCreateInfo raster;
		VkPipelineColorBlendAttachmentState    colorBlendAttachment;
		VkPipelineMultisampleStateCreateInfo   multisampling;
		VkPipelineDepthStencilStateCreateInfo  depthStencil;

		VkPipelineRenderingCreateInfo          renderInfo;
		VkPipelineLayout					   pipelineLayout;
		VkFormat                               colorAttachmentFormat;

		static void create(GraphicsPipelineBuilder* builder);

		void set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader);
		void set_topology(VkPrimitiveTopology topology);
		void set_cull_mode(VkCullModeFlags cullMode);
		void set_polygon_mode(VkPolygonMode polyMode);
		void set_color_attachment_format(VkFormat format);
		void set_depth_format(VkFormat format);
		void disable_depth_test();
		void clear();

		VkPipeline build_pipeline(VkDevice device);
	};

	// Loads SPIR-V shader
	bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);
	
};