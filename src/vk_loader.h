#pragma once

#include <vk_types.h>
#include <unordered_map>
#include <filesystem>

struct MeshSurface
{
	uint32_t startIndex;
	uint32_t count;
};

struct Mesh
{
	std::string name;

	std::vector<MeshSurface> surfaces;
	GPUMeshBuffers		     meshBuffers;
};


struct DeletionQueue
{
	std::vector<std::function<void()>> queue;

	void push(std::function<void()> func);

	void flush();
};

class VulkanEngine;

std::optional<std::vector<std::shared_ptr<Mesh>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);