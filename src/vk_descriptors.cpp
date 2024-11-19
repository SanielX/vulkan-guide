#include <vk_descriptors.h>

void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type)
{
	VkDescriptorSetLayoutBinding bind{};
	bind.binding = binding;
	bind.descriptorCount = 1;
	bind.descriptorType  = type;

	bindings.push_back(bind);
}

void DescriptorLayoutBuilder::clear()
{
	bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags stages, void* pNext, VkDescriptorSetLayoutCreateFlags flags)
{
	for (auto& b : bindings)
	{
		b.stageFlags |= stages;
	}

	VkDescriptorSetLayoutCreateInfo info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = pNext };
	info.pBindings    = bindings.data();
	info.bindingCount = (uint32_t)bindings.size();
	info.flags		  = flags;

	VkDescriptorSetLayout set;
	vkCreateDescriptorSetLayout(device, &info, nullptr, &set);

	return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
	std::vector<VkDescriptorPoolSize> poolSizes;
	for (PoolSizeRatio ratio : poolRatios)
	{
		poolSizes.push_back(VkDescriptorPoolSize{ .type = ratio.type, .descriptorCount = uint32_t(ratio.ratio * maxSets) });
	}

	VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr };
	pool_info.flags   = 0;
	pool_info.maxSets = maxSets;
	pool_info.poolSizeCount = poolSizes.size();
	pool_info.pPoolSizes = poolSizes.data();

	vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device)
{
	vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device)
{
	vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
	VkDescriptorSetAllocateInfo allocInfo = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr };
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet ds;
	VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

	return ds;
}

void DescriptorAllocatorGrowable::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
	ratios.clear();
	for (PoolSizeRatio r : poolRatios)
	{
		ratios.push_back(r);
	}

	VkDescriptorPool newPool = create_pool(device, maxSets, poolRatios);
	setsPerPool = (int)(maxSets * 1.5);

	readyPools.push_back(newPool);
}

void DescriptorAllocatorGrowable::clear_descriptors(VkDevice device)
{
	for (VkDescriptorPool p : readyPools)
	{
		vkResetDescriptorPool(device, p, 0);
	}
	for (VkDescriptorPool p : fullPools)
	{
		vkResetDescriptorPool(device, p, 0);
		readyPools.push_back(p);
	}

	fullPools.clear();
}

void DescriptorAllocatorGrowable::destroy_pool(VkDevice device)
{
	for (VkDescriptorPool p : readyPools)
	{
		vkDestroyDescriptorPool(device, p, 0);
	}
	for (VkDescriptorPool p : fullPools)
	{
		vkDestroyDescriptorPool(device, p, 0);
	}

	readyPools.clear();
	fullPools.clear();
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
	VkDescriptorPool poolToUse = get_pool(device);

	VkDescriptorSetAllocateInfo allocInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	allocInfo.descriptorPool	 = poolToUse;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts		 = &layout;

	VkDescriptorSet ds;
	VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &ds);

	if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
	{
		fullPools.push_back(poolToUse);
		poolToUse = get_pool(device);
		allocInfo.descriptorPool = poolToUse;

		// Try again, if we fail then well, screw it all!
		VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));
	}

	readyPools.push_back(poolToUse);
	return ds;
}

VkDescriptorPool DescriptorAllocatorGrowable::get_pool(VkDevice device)
{
	VkDescriptorPool newPool;
	if (readyPools.size() != 0)
	{
		newPool = readyPools.back();
		readyPools.pop_back();
	}
	else
	{
		newPool = create_pool(device, setsPerPool, ratios);

		setsPerPool = std::min(4096, (int)(setsPerPool * 1.5));
	}

	return newPool;
}

VkDescriptorPool DescriptorAllocatorGrowable::create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios)
{
	std::vector<VkDescriptorPoolSize> poolSizes;
	for (PoolSizeRatio ratio : poolRatios)
	{
		poolSizes.push_back(VkDescriptorPoolSize{
			.type = ratio.type,
			.descriptorCount = uint32_t(ratio.ratio * setCount)
		});
	}

	VkDescriptorPoolCreateInfo pool_info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr };
	pool_info.flags = 0;
	pool_info.maxSets = setCount;
	pool_info.poolSizeCount = (uint32_t)poolSizes.size();
	pool_info.pPoolSizes = poolSizes.data();

	VkDescriptorPool newPool;
	vkCreateDescriptorPool(device, &pool_info, nullptr, &newPool);
	return newPool;
}

void DescriptorWriter::write_image(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type)
{
	VkDescriptorImageInfo& info = imageInfos.emplace_back(VkDescriptorImageInfo{
		.sampler     = sampler,
		.imageView   = image,
		.imageLayout = layout,
		});

	VkWriteDescriptorSet writeop = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeop.dstBinding		= binding;
	writeop.descriptorCount = 1;
	writeop.descriptorType  = type;
	writeop.pImageInfo		= &info;

	writeop.dstSet = VK_NULL_HANDLE; // set later in update_set
	writes.push_back(writeop);
}

void DescriptorWriter::write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type)
{
	VkDescriptorBufferInfo& info = bufferInfos.emplace_back(VkDescriptorBufferInfo{
			.buffer = buffer,
			.offset = offset,
			.range  = size,
		});

	VkWriteDescriptorSet writeop = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeop.dstBinding		= binding;
	writeop.descriptorCount = 1;
	writeop.descriptorType  = type;
	writeop.pBufferInfo     = &info;

	writeop.dstSet = VK_NULL_HANDLE; // set later in update_set
	writes.push_back(writeop);
}

void DescriptorWriter::clear()
{
	imageInfos.clear();
	writes.clear();
	bufferInfos.clear();
}

void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set)
{
	for (VkWriteDescriptorSet& write : writes)
	{
		write.dstSet = set;
	}

	vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}
