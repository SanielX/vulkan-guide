#include <vk_pipelines.h>
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

void vkutil::GraphicsPipelineBuilder::create(GraphicsPipelineBuilder* builder)
{
	*builder = {};
	builder->clear();
}

void vkutil::GraphicsPipelineBuilder::set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader)
{
	shaderStages.clear();
	shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT,   vertexShader));
	shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));
}

void vkutil::GraphicsPipelineBuilder::set_topology(VkPrimitiveTopology topology)
{
	inputAssembly.topology = topology;

	// primitiveRestartEnable member to VK_TRUE, then it's possible to break up 
	// lines and triangles in the _STRIP topology modes by using a special index of 0xFFFF or 0xFFFFFFFF.
	inputAssembly.primitiveRestartEnable = VK_FALSE;  
}

void vkutil::GraphicsPipelineBuilder::set_cull_mode(VkCullModeFlags cullMode)
{
	raster.cullMode = cullMode;
}

void vkutil::GraphicsPipelineBuilder::set_polygon_mode(VkPolygonMode polyMode)
{
	raster.polygonMode = polyMode;
	raster.lineWidth   = 1.0f;
}

void vkutil::GraphicsPipelineBuilder::set_color_attachment_format(VkFormat format)
{
	colorAttachmentFormat = format;

	renderInfo.colorAttachmentCount    = 1;
	renderInfo.pColorAttachmentFormats = &colorAttachmentFormat;
}

void vkutil::GraphicsPipelineBuilder::set_depth_format(VkFormat format)
{
	renderInfo.depthAttachmentFormat = format;
}

void vkutil::GraphicsPipelineBuilder::disable_depth_test()
{
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.depthWriteEnable      = VK_FALSE;
	depthStencil.depthCompareOp        = VK_COMPARE_OP_NEVER;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable	   = VK_FALSE;	
	depthStencil.front = {}; // stencil OP
	depthStencil.back  = {};
	depthStencil.minDepthBounds = 0;
	depthStencil.maxDepthBounds = 1;
}

void vkutil::GraphicsPipelineBuilder::set_depth_test()
{
	depthStencil.depthTestEnable       = VK_TRUE;
	depthStencil.depthCompareOp		   = VK_COMPARE_OP_GREATER_OR_EQUAL;
	depthStencil.depthWriteEnable      = VK_TRUE;
	depthStencil.depthBoundsTestEnable = VK_TRUE;

	depthStencil.minDepthBounds = 0.f;
	depthStencil.maxDepthBounds = 1.f;
	set_cull_mode(VK_CULL_MODE_FRONT_BIT);
}

void vkutil::GraphicsPipelineBuilder::clear()
{
	inputAssembly        = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
	
	raster		         = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
	colorBlendAttachment = {};
	multisampling        = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
	pipelineLayout       = {};
	depthStencil         = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	renderInfo           = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };

	shaderStages.clear();

	raster.frontFace = VK_FRONT_FACE_CLOCKWISE;

	multisampling.sampleShadingEnable   = VK_FALSE;
	multisampling.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading      = 1.0f;
	multisampling.pSampleMask           = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable      = VK_FALSE;

	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable    = false;

	set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	set_polygon_mode(VK_POLYGON_MODE_FILL);
	set_cull_mode(VK_CULL_MODE_BACK_BIT);
}

VkPipeline vkutil::GraphicsPipelineBuilder::build_pipeline(VkDevice device)
{
	VkPipelineViewportStateCreateInfo viewportState{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = nullptr };
	viewportState.viewportCount = 1;
	viewportState.scissorCount  = 1;

	VkPipelineColorBlendStateCreateInfo colorBlending{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .pNext = nullptr };
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp		= VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nullptr };

	VkGraphicsPipelineCreateInfo pipelineInfo{ .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = nullptr };
	pipelineInfo.pNext = &renderInfo;
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pStages = shaderStages.data();

	pipelineInfo.pVertexInputState   = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState		 = &viewportState;
	pipelineInfo.pRasterizationState = &raster;
	pipelineInfo.pMultisampleState   = &multisampling;
	pipelineInfo.pColorBlendState    = &colorBlending;
	pipelineInfo.pDepthStencilState  = &depthStencil;
	pipelineInfo.layout				 = pipelineLayout;

	VkDynamicState state[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .pNext = nullptr };
	dynamicInfo.pDynamicStates    = state;
	dynamicInfo.dynamicStateCount = std::size(state);

	pipelineInfo.pDynamicState = &dynamicInfo;

	VkPipeline pipeline;
	bool result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
	if (result != VK_SUCCESS) 
	{
		fmt::print("[ERROR] Failed to created Vulkan pipeline");
		return VK_NULL_HANDLE;
	}

	return pipeline;
}
