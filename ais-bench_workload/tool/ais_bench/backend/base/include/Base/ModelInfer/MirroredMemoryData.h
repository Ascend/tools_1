#ifndef _MIRRORED_MEMORY_DATA_H
#define _MIRRORED_MEMORY_DATA_H

#include <vector>

#include "Base/Log/Log.h"
#include "Base/MemoryHelper/MemoryHelper.h"

namespace Base {

struct MirroredMemoryData {
    std::vector<MemoryData> host_mem;
    std::vector<MemoryData> device_mem;

    APP_ERROR AllocateInputMemory(std::vector<BaseTensor> &inputs_host, std::vector<BaseTensor> &inputs_device, int deviceId_) {
        for (auto &feed : inputs_host) {
            Base::MemoryData src(feed.buf, feed.size, MemoryData::MemoryType::MEMORY_HOST, deviceId_);
            Base::MemoryData dst(feed.size, MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
            MemoryHelper::MxbsMalloc(dst);
            device_mem.push_back(dst);
            host_mem.push_back(src);
            inputs_device.emplace_back(dst.ptrData, dst.size);
        }
        return APP_ERR_OK;
    }

    APP_ERROR AllocateOutputMemory(std::vector<BaseTensor> &outputs_device,
                                   std::vector<std::pair<std::string, size_t>> output_desc,
                                   std::vector<size_t> customOutputSizes, int device_id) {
        size_t size;
        size_t customIndex = 0;
        std::vector<MemoryData> outputs;
        for (size_t i = 0; i < output_desc.size(); ++i) {
            size = output_desc[i].second;
            std::string name = output_desc[i].first;
            if (customIndex < customOutputSizes.size()){
                size = customOutputSizes[customIndex++];
            }
            if (size == 0){
                ERROR_LOG("out i:%zu size is zero", i);
                return APP_ERR_INFER_OUTPUTSIZE_IS_ZERO;
            }
            DEBUG_LOG("Create OutMemory i:%zu name:%s size:%zu", i, name.c_str(), size);
            Base::MemoryData memory_device(size, MemoryData::MemoryType::MEMORY_DEVICE, device_id);
            auto ret = MemoryHelper::MxbsMalloc(memory_device);
            if (ret != APP_ERR_OK) {
                ERROR_LOG("MemoryHelper::MxbsMalloc failed.i:%zu name:%s size:%zu ret:%d", \
                        i, name.c_str(), size, ret);
                return ret;
            }
            Base::MemoryData memory_host(size);
            ret = MemoryHelper::MxbsMalloc(memory_host);
            if (ret != APP_ERR_OK) {
                ERROR_LOG("MemoryHelper::MxbsMalloc failed.i:%zu name:%s size:%zu ret:%d", \
                        i, name.c_str(), size, ret);
                return ret;
            }
            outputs_device.emplace_back(memory_device.ptrData, memory_device.size);
            device_mem.push_back(std::move(memory_device));
            host_mem.push_back(std::move(memory_host));
        }

        return APP_ERR_OK;
    }

    APP_ERROR FreeInputMemory() {
        for (auto &memData : device_mem) {
            MemoryHelper::MxbsFree(memData);
        }
        return APP_ERR_OK;
    }

    APP_ERROR FreeOutputMemory() {
        for (auto &memData : device_mem) {
            MemoryHelper::MxbsFree(memData);
        }
        for (auto &memData : host_mem) {
            MemoryHelper::MxbsFree(memData);
        }
        return APP_ERR_OK;
    }
};

} // !namespace Base

#endif
