// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include "vk_descriptors.h"
#include "vk_loader.h"

constexpr unsigned int FRAME_OVERLAP = 2;

struct FrameData
{
	VkCommandPool   command_pool;
	VkCommandBuffer cmd;

	VkSemaphore swapchain_semaphore;
	VkSemaphore render_semaphore;

	VkFence		render_fence;
};

struct ComputePushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeShader
{
	const char* name;

	VkPipeline		 pipeline;
	VkPipelineLayout pipeline_layout;

	ComputePushConstants data;
};

class VulkanEngine {
public:

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

	//run main loop
	void run();

	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_mesh_pipelines();
	void init_triangle_pipeline();
	void init_imgui();

	void create_swapchain(uint32_t w, uint32_t h);
	void destroy_swapchain();

	GraphicsBuffer create_graphics_buffer(size_t allocSize, VkBufferUsageFlags usageFlags, VmaMemoryUsage memoryUsage);
	void		   destroy_graphics_buffer(GraphicsBuffer *buffer);

	GPUMeshBuffers create_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	void immediate_submit(std::function<void(VkCommandBuffer cmd)> && function);

	void draw_hello_triangle(VkCommandBuffer cmd);

	DescriptorAllocator descriptorAllocator;

	VkPipelineLayout trianglePipelineLayout;
	VkPipeline trianglePipeline;

	VkPipelineLayout meshPipelineLayout;
	VkPipeline       meshPipeline;
	GPUMeshBuffers   rectangle;

	std::vector<std::shared_ptr<Mesh>> testMeshes;
	int   currentMesh = 0;
	float meshRotateAngle = 0;
	glm::vec3 meshPosition = {};
	glm::vec3 meshScale    = {1,1,1};
	glm::mat4 modelMatrix;

	VkDescriptorSet	      drawImageDescriptorSet;
	VkDescriptorSetLayout drawImageDescriptorLayout;

	VkPipeline		 gradientPipeline;
	VkPipelineLayout gradientPipelineLayout;

	VkFence			imFence;
	VkCommandBuffer imCommandBuffer;
	VkCommandPool   imCommandPool;

	std::vector<ComputeShader> backgroundShaders;
	int currentBackgroundShader = 0;

	static VulkanEngine& Get();

	bool       _isInitialized = false;
	int        _frameNumber   = 0;
	bool       stop_rendering = false;
	VkExtent2D _windowExtent  = { 1280 , 720 };

	VkInstance vk_instance;
	VkDebugUtilsMessengerEXT debug_messanger;
	VkPhysicalDevice         physical_device;
	VkDevice				 device;
	VkSurfaceKHR             surface;

	VmaAllocator allocator;

	VkSwapchainKHR swapchain;
	VkFormat	   swapchain_format;

	std::vector<VkImage>	 swapchain_images;
	std::vector<VkImageView> swapchain_image_views;
	VkExtent2D				 swapchain_extent;

	Texture    drawImage;
	Texture    drawDepthBuffer;
	VkExtent2D drawExtent;

	FrameData	 frames[FRAME_OVERLAP];
	unsigned int frame_count;

	VkQueue  graphics_queue;
	uint32_t graphics_queue_family;

	FrameData& get_current_frame() { return frames[frame_count % FRAME_OVERLAP]; }

	struct SDL_Window* _window{ nullptr };
};
