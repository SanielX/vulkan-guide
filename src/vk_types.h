// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <glm/common.hpp>

struct Texture
{
    VkImage       image;
    VkImageView   imageView;
    VmaAllocation allocation;
    VkExtent3D    extent;
    VkFormat      format;
};

struct GraphicsBuffer
{
    VkBuffer          buffer;
    VmaAllocation     allocation;
    VmaAllocationInfo allocInfo;
};

struct Vertex
{
    glm::vec3 position;
    float uv_x;
    
    glm::vec3 normal;
    float uv_y;

    glm::vec4 color;
};

struct GPUMeshBuffers
{
    GraphicsBuffer  indexBuffer;
    GraphicsBuffer  vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

struct MeshDrawPushConstants
{
    glm::mat4       worldMatrix;
    VkDeviceAddress vertexBuffer;
};

struct DrawContext;

class IRenderable
{
    virtual void Draw(const glm::mat4& matrix, DrawContext& ctx) = 0;
};

struct Node : public IRenderable
{
    std::weak_ptr<Node>                parent;
    std::vector<std::shared_ptr<Node>> children;

    glm::mat4 localTransform;
    glm::mat4 worldTransform;

    void refreshTransform(const glm::mat4& parentMatrix);
    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx);
};


#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)
