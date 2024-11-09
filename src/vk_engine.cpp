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
    vkCreatePipelineLayout(device, &computeLayout, nullptr, &gradientPipelineLayout);

    // Load compute shader
    VkShaderModule computeDrawShader;
    bool result = vkutil::load_shader_module("../../shaders/gradient.comp.spv", device, &computeDrawShader);
    if (!result) 
    {
        fmt::print("Error while loading compute shader gradient.comp");
        exit(-100);
    }
    // Create shader stage from spir-v file by setting shader type and entry function name
    VkPipelineShaderStageCreateInfo stageInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = nullptr };
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = computeDrawShader;
    stageInfo.pName  = "main";
    // Create pipeline, which is a combo of a shader and pipeline layout (descriptors+constants)
    VkComputePipelineCreateInfo computePipelineCreateInfo{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = nullptr };
    computePipelineCreateInfo.layout = gradientPipelineLayout;
    computePipelineCreateInfo.stage  = stageInfo;

    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradientPipeline);

    /* Pretend to clean up
    vkDestroyShaderModule(_device, computeDrawShader, nullptr);

	_mainDeletionQueue.push_function([&]() {
		vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
		vkDestroyPipeline(_device, _gradientPipeline, nullptr);
		});
    */
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
    vkutil::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // VkClearColorValue clearValue;
    // float flash = std::abs(std::sin(_frameNumber / 120.f));
    // clearValue = { { 0.0f, 0.0f, flash, 1.0f } };
    // VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    // vkCmdClearColorImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gradientPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gradientPipelineLayout, /*first set*/ 0, /*set count*/1, &drawImageDescriptorSet, 0, nullptr);
    vkCmdDispatch(cmd, std::ceil(drawExtent.width / 16.0), std::ceil(drawExtent.height / 16.0), 1);

    vkutil::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL,   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, swapchainTarget, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkutil::blit_image(cmd, drawImage.image, swapchainTarget, drawExtent, swapchain_extent);

    vkutil::transition_image(cmd, swapchainTarget, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
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
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}