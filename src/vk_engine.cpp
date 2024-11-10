﻿//> includes
#define VMA_IMPLEMENTATION
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <thread>

#include "VkBootstrap.h"
#include "vk_images.h"
#include "vk_pipelines.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow(
        "VkGuide App",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();
    init_pipelines();
    init_imgui();
    init_triangle_pipeline();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;
    auto build_result = builder.set_app_name("VkGuide App")
                        .request_validation_layers()
                        .use_default_debug_messenger()
                        .require_api_version(1, 3, 0)
                        .build();

    if (!build_result.has_value())
    {
        fmt::print("{}", build_result.error().message());
        exit(build_result.error().value());
    }

    vkb::Instance vkb_instance = build_result.value();
    vk_instance     = vkb_instance.instance;
    debug_messanger = vkb_instance.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, vk_instance, &surface);

    VkPhysicalDeviceVulkan13Features features13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing  = true;

    vkb::PhysicalDeviceSelector selector{ vkb_instance };
    vkb::PhysicalDevice vkb_physical_device = selector
                                     .set_minimum_version(1, 3)
                                     .set_required_features_13(features13)
                                     .set_required_features_12(features12)
                                     .set_surface(surface).select().value();

    vkb::DeviceBuilder device_builder{ vkb_physical_device };
    vkb::Device        vkb_device = device_builder.build().value();

    device          = vkb_device.device;
    physical_device = vkb_physical_device.physical_device;

    graphics_queue        = vkb_device.get_queue(vkb::QueueType::graphics).value();
    graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = physical_device;
    allocatorInfo.device = device;
    allocatorInfo.instance = vk_instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &allocator);
    // TODO: Delete allocator
}

void VulkanEngine::init_swapchain()
{
    int w, h;
    SDL_Vulkan_GetDrawableSize(_window, &w, &h);
    fmt::print("Initialize Swapchain, width: {}, height: {}", w, h);

    create_swapchain(w, h);

    VkExtent3D drawImageExtent = { w, h, 1 };
    drawImage.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    drawImage.extent = drawImageExtent;

    VkImageUsageFlags drawImageUsageFlags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |    // Can copy from this image
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |    // Can copy into this image
                                            VK_IMAGE_USAGE_STORAGE_BIT      |    // Can write to in compute shaders
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // Can bind as raster draw target
    VkImageCreateInfo img_info = vkinit::image_create_info(drawImage.format, drawImageUsageFlags, drawImageExtent);

    VmaAllocationCreateInfo img_allocInfo{};
    img_allocInfo.usage         = VMA_MEMORY_USAGE_GPU_ONLY;
    img_allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(allocator, &img_info, &img_allocInfo, &drawImage.image, &drawImage.allocation, nullptr);

    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(drawImage.format, drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
    vkCreateImageView(device, &view_info, nullptr, &drawImage.imageView);

    drawExtent = { .width = (uint32_t)w, .height = (uint32_t)h };
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.pNext = nullptr;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmdPoolInfo.queueFamilyIndex = graphics_queue_family;

    for (size_t i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &frames[i].command_pool));

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.pNext = nullptr;
        cmdAllocInfo.commandPool = frames[i].command_pool;
        cmdAllocInfo.commandBufferCount = 1;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &frames[i].cmd));
    }

    // ------>8------Init immediate buffers

    vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &imCommandPool);

    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(imCommandPool, 1);
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &imCommandBuffer);
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo     fenceCreateInfo     = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (size_t i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frames[i].render_fence));

        VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].render_semaphore));
        VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].swapchain_semaphore));
    }

    vkCreateFence(device, &fenceCreateInfo, nullptr, &imFence);
}

void VulkanEngine::init_descriptors()
{
    // Create a layout descriptor
    // E.g. at index 0 there is a storage image (RWTexture2D register(0))
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    drawImageDescriptorLayout = layoutBuilder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);

    // Create pool of descriptors we can allocate from. 
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = { { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 } };
    descriptorAllocator.init_pool(device, 10, sizes);

    // Allocate actual descriptor. Or type image will hold ptr to the image and view (size, mip levels etc)
    drawImageDescriptorSet = descriptorAllocator.allocate(device, drawImageDescriptorLayout);

    // Populate the descriptor
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView   = drawImage.imageView;
    
    // Create a write operation,
    VkWriteDescriptorSet drawImageWrite{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr };
    drawImageWrite.dstSet          = drawImageDescriptorSet; // for given descriptor set
    drawImageWrite.dstBinding      = 0;                      // at given binding   
    drawImageWrite.descriptorCount = 1;                      // We set 1 descriptor
    drawImageWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; // of type image
    drawImageWrite.pImageInfo      = &imageInfo;                       // and because of that we pass 1 imageInfo there

    vkUpdateDescriptorSets(device, 1, &drawImageWrite, 0, nullptr);
    /* globalDescriptorAllocator.destroy_pool(_device);

		vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr); */
}

void VulkanEngine::init_pipelines()
{
    // Create pipeline layout that has only 1 descriptor set and no push constants
    VkPipelineLayoutCreateInfo computeLayout{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = nullptr };
    computeLayout.pSetLayouts    = &drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;
     
    VkPushConstantRange pushConstants{};
    pushConstants.offset = 0;
    pushConstants.size = sizeof(ComputePushConstants);
    pushConstants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstants;
    computeLayout.pushConstantRangeCount = 1;

    vkCreatePipelineLayout(device, &computeLayout, nullptr, &gradientPipelineLayout);

    // Load compute shader
    VkShaderModule gradientModule;
    bool result = vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", device, &gradientModule);
    if (!result) 
    {
        fmt::print("Error while loading compute shader gradient.comp");
        exit(-100);
    }

    VkShaderModule skyModule;
    /*bool*/ result = vkutil::load_shader_module("../../shaders/sky.comp.spv", device, &skyModule);
    if (!result)
    {
        fmt::print("Error while loading compute shader sky.comp");
        exit(-100);
    }

    // Create shader stage from spir-v file by setting shader type and entry function name
    VkPipelineShaderStageCreateInfo stageInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = nullptr };
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientModule;
    stageInfo.pName  = "main";

    // Create pipeline, which is a combo of a shader and pipeline layout (descriptors+constants)
    VkComputePipelineCreateInfo computePipelineCreateInfo{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = nullptr };
    computePipelineCreateInfo.layout = gradientPipelineLayout;
    computePipelineCreateInfo.stage  = stageInfo;

    ComputeShader gradient;
    gradient.pipeline_layout = gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};
    gradient.data.data1 = glm::vec4(1, 1, 0, 0);
    gradient.data.data2 = glm::vec4(0, 1, 1, 0);
    
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline);

    computePipelineCreateInfo.stage.module = skyModule;

    ComputeShader sky;
    sky.pipeline_layout = gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4f, 0.97f);

    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline);

    backgroundShaders.push_back(gradient);
    backgroundShaders.push_back(sky);

    vkDestroyShaderModule(device, gradientModule, nullptr);
    vkDestroyShaderModule(device, skyModule,      nullptr);

    /* Pretend to clean up
	_mainDeletionQueue.push_function([&]() {
		vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
		vkDestroyPipeline(_device, _gradientPipeline, nullptr);
		});
    */
}

void VulkanEngine::init_triangle_pipeline()
{
    VkShaderModule fragModule;
    bool result = vkutil::load_shader_module("../../shaders/colored_triangle.frag.spv", device, &fragModule);
    if (!result) throw std::exception("Failed to load colored_triangle.frag");

    VkShaderModule vertModule;
    result = vkutil::load_shader_module("../../shaders/colored_triangle.vert.spv", device, &vertModule);
    if (!result) throw std::exception("Failed to load colored_triangle.vert.spv");

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &trianglePipelineLayout);

    vkutil::GraphicsPipelineBuilder builder;
    vkutil::GraphicsPipelineBuilder::create(&builder);

    builder.pipelineLayout = trianglePipelineLayout;
    builder.set_shaders(vertModule, fragModule);
    builder.set_cull_mode(VK_CULL_MODE_BACK_BIT);
    builder.disable_depth_test();
    builder.set_color_attachment_format(drawImage.format);
    builder.set_depth_format(VK_FORMAT_UNDEFINED);
    builder.set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    trianglePipeline = builder.build_pipeline(device);

    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);
    /* pretend to clean up
    _mainDeletionQueue.push_function([&]() {
		vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
		vkDestroyPipeline(_device, _trianglePipeline, nullptr);
	});
    */
}

void VulkanEngine::init_imgui()
{
    VkDescriptorPoolSize pool_sizes[] = { 
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

    VkDescriptorPoolCreateInfo pool_info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr };
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes    = pool_sizes;

    VkDescriptorPool imguiPool;
    vkCreateDescriptorPool(device, &pool_info, nullptr, &imguiPool);

    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForVulkan(_window);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = vk_instance;
    init_info.PhysicalDevice = physical_device;
    init_info.Device = device;
    init_info.Queue = graphics_queue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;
 //   init_info.Allocator = &allocator->m_AllocationCallbacks;

    init_info.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain_format;
    
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);
    ImGui_ImplVulkan_CreateFontsTexture();
    /* Pretend to clean up
    ImGui_ImplVulkan_Shutdown();
	vkDestroyDescriptorPool(_device, imguiPool, nullptr);*/
}


void VulkanEngine::cleanup()
{
    if (_isInitialized) 
    {
        vkDeviceWaitIdle(device);

        for (size_t i = 0; i < FRAME_OVERLAP; i++)
        {
            vkDestroyCommandPool(device, frames[i].command_pool, nullptr);
        }

        destroy_swapchain();
        
        vkDestroySurfaceKHR(vk_instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);

        vkb::destroy_debug_utils_messenger(vk_instance, debug_messanger);
        vkDestroyInstance(vk_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::create_swapchain(uint32_t w, uint32_t h)
{
    vkb::SwapchainBuilder builder{ physical_device, device, surface };
    swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;

    VkSurfaceFormatKHR format = VkSurfaceFormatKHR{ .format = swapchain_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    vkb::Swapchain vkb_swapchain = builder.set_desired_format(format)
                                   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                                   .set_desired_extent(w,h)
                                   .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT) 
                                   .build().value();

    swapchain_extent      = vkb_swapchain.extent;
    swapchain             = vkb_swapchain.swapchain;
    swapchain_images      = vkb_swapchain.get_images().value();
    swapchain_image_views = vkb_swapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    for (size_t i = 0; i < swapchain_images.size(); i++)
    {
        vkDestroyImageView(device, swapchain_image_views[i], nullptr);
    }
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    vkResetFences(device, 1, &imFence);
    vkResetCommandBuffer(imCommandBuffer, 0);

    VkCommandBuffer cmd = imCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    vkBeginCommandBuffer(cmd, &cmdBeginInfo);

    function(cmd);

    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, /*signal*/ nullptr, /*wait*/ nullptr);

    vkQueueSubmit2(graphics_queue, 1, &submit, imFence);

    vkWaitForFences(device, 1, &imFence, true, 9999999999);
}

void VulkanEngine::draw_hello_triangle(VkCommandBuffer cmd)
{
    // Info about image attachment (image view + current layout + load/store OP)
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Defines which color+depth+stencil images are bounds to the graphics pipeline
    VkRenderingInfo renderInfo = vkinit::rendering_info(drawExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);

    VkViewport viewport{};
    viewport.width  = drawExtent.width;
    viewport.height = drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent.width  = drawExtent.width;
    scissor.extent.height = drawExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // cmd, vertex count, instance count, first vertex, first index
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw()
{
    const uint32_t one_second = 1000000000;
    FrameData& frame = get_current_frame();

    VK_CHECK(vkWaitForFences(device, 1, &frame.render_fence, true, one_second));
    VK_CHECK(vkResetFences  (device, 1, &frame.render_fence));

    uint32_t swapchainImageIndex;
    vkAcquireNextImageKHR(device, swapchain, one_second, frame.swapchain_semaphore, nullptr, &swapchainImageIndex);

    VkCommandBuffer cmd = frame.cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    vkBeginCommandBuffer(cmd, &cmdBeginInfo);



    // --- CMD BEGIN ------------------------------------------------------------------------------------------------------------------
    VkImage& swapchainTarget = swapchain_images[swapchainImageIndex];
    VkImageView& swapchainTargetView = swapchain_image_views[swapchainImageIndex];
    vkutil::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // VkClearColorValue clearValue;
    // float flash = std::abs(std::sin(_frameNumber / 120.f));
    // clearValue = { { 0.0f, 0.0f, flash, 1.0f } };
    // VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    // vkCmdClearColorImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

    ComputeShader& currentShader = backgroundShaders[currentBackgroundShader];

    vkCmdBindPipeline      (cmd, VK_PIPELINE_BIND_POINT_COMPUTE, currentShader.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, currentShader.pipeline_layout, /*first set*/ 0, /*set count*/1, &drawImageDescriptorSet, 0, nullptr);

    vkCmdPushConstants(cmd, gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &currentShader.data);

    vkCmdDispatch(cmd, std::ceil(drawExtent.width / 16.0), std::ceil(drawExtent.height / 16.0), 1);

    vkutil::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    draw_hello_triangle(cmd);

    vkutil::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, swapchainTarget, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkutil::blit_image(cmd, drawImage.image, swapchainTarget, drawExtent, swapchain_extent);

    vkutil::transition_image(cmd, swapchainTarget, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    draw_imgui(cmd, swapchainTargetView);

    vkutil::transition_image(cmd, swapchainTarget, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    // --- CMD END ------------------------------------------------------------------------------------------------------------------
        
    VK_CHECK(vkEndCommandBuffer(cmd));

    //prepare the submission to the queue. 
    //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //we will signal the _renderSemaphore, to signal that rendering has finished

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);

    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame().swapchain_semaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame().render_semaphore);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

    //submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, frame.render_fence));

    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that, 
    // as its necessary that drawing commands have finished before the image is displayed to the user
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &get_current_frame().render_semaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VK_CHECK(vkQueuePresentKHR(graphics_queue, &presentInfo));

    //increase the number of frames drawn
    _frameNumber++;
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(swapchain_extent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }

            if (e.type == SDL_KEYDOWN)
            {
                auto key_name = SDL_GetKeyName(e.key.keysym.sym);
                fmt::print("Key press: {}\n", key_name);
            }

            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::ShowDemoWindow();
        
        ImGui::Begin("Background");

        ImGui::SliderInt("Effect", &currentBackgroundShader, 0, 1);
        ImGui::ColorEdit4("Color 0", &backgroundShaders[currentBackgroundShader].data.data1.x);
        ImGui::ColorEdit4("Color 1", &backgroundShaders[currentBackgroundShader].data.data2.x);
        ImGui::ColorEdit4("Color 2", &backgroundShaders[currentBackgroundShader].data.data3.x);
        ImGui::ColorEdit4("Color 3", &backgroundShaders[currentBackgroundShader].data.data4.x);

        ImGui::End();

        ImGui::Render();

        draw();
    }
}