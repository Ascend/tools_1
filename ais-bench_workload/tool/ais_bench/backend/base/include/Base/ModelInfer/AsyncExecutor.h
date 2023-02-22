#ifndef _ASYNC_EXECUTOR_H
#define _ASYNC_EXECUTOR_H

#include <stdexcept>
#include <vector>

#include "acl/acl.h"

#include "Base/DeviceManager/DeviceManager.h"
#include "Base/ErrorCode/ErrorCode.h"
#include "Base/Log/Log.h"
#include "Base/MemoryHelper/MemoryHelper.h"
#include "MirroredMemoryData.h"
#include "ModelInferenceProcessor.h"

namespace Base {

struct EventGroup {
    aclrtEvent INPUT_S{}, INPUT_E{};
    aclrtEvent COMPUTE_S{}, COMPUTE_E{};
    aclrtEvent OUTPUT_S{}, OUTPUT_E{};

    EventGroup(const EventGroup &) = delete;
    EventGroup &operator=(const EventGroup &) = delete;

    EventGroup(EventGroup &&rhs) noexcept {
        *this = std::move(rhs);
    }

    EventGroup &operator=(EventGroup &&rhs) noexcept {
        INPUT_S = rhs.INPUT_S;
        INPUT_E = rhs.INPUT_E;
        COMPUTE_S = rhs.COMPUTE_S;
        COMPUTE_E = rhs.COMPUTE_E;
        OUTPUT_S = rhs.OUTPUT_S;
        OUTPUT_E = rhs.OUTPUT_E;
        rhs.INPUT_S = nullptr;
        rhs.INPUT_E = nullptr;
        rhs.COMPUTE_S = nullptr;
        rhs.COMPUTE_E = nullptr;
        rhs.OUTPUT_S = nullptr;
        rhs.OUTPUT_E = nullptr;
        return *this;
    }

    EventGroup() {
        aclError ret = aclrtCreateEvent(&INPUT_S);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
        ret = aclrtCreateEvent(&INPUT_E);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
        ret = aclrtCreateEvent(&COMPUTE_S);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
        ret = aclrtCreateEvent(&COMPUTE_E);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
        ret = aclrtCreateEvent(&OUTPUT_S);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
        ret = aclrtCreateEvent(&OUTPUT_E);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
    }

    ~EventGroup() {
        if (INPUT_S != nullptr) {
            aclrtDestroyEvent(INPUT_S);
        }
        if (INPUT_E != nullptr) {
            aclrtDestroyEvent(INPUT_E);
        }
        if (COMPUTE_S != nullptr) {
            aclrtDestroyEvent(COMPUTE_S);
        }
        if (COMPUTE_E != nullptr) {
            aclrtDestroyEvent(COMPUTE_E);
        }
        if (OUTPUT_S != nullptr) {
            aclrtDestroyEvent(OUTPUT_S);
        }
        if (OUTPUT_E != nullptr) {
            aclrtDestroyEvent(OUTPUT_E);
        }
    }
};

class AsyncExecutor {
  public:
    AsyncExecutor(const AsyncExecutor &) = delete;
    AsyncExecutor(AsyncExecutor &&) = delete;
    AsyncExecutor &operator=(const AsyncExecutor &) = delete;
    AsyncExecutor &operator=(AsyncExecutor &&) = delete;

    explicit AsyncExecutor(int depth = 2)
        : depth(depth), active(depth), eventGroups(depth) {
        aclError ret = aclrtCreateStream(&stream_input);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
        ret = aclrtCreateStream(&stream_compute);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
        ret = aclrtCreateStream(&stream_output);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
    }

    ~AsyncExecutor() {
        aclrtDestroyStream(stream_input);
        aclrtDestroyStream(stream_compute);
        aclrtDestroyStream(stream_output);
    }

    void Sync() {
        if (cur < active.size() && cur < eventGroups.size() && active[cur] != 0) {
            EventGroup &eventGroup = eventGroups[cur];
            aclError ret = ACL_SUCCESS;
            ret = aclrtSynchronizeEvent(eventGroup.OUTPUT_E);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtSynchronizeEvent failed.");
            }
            ret = aclrtResetEvent(eventGroup.INPUT_S, stream_input);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(aclGetRecentErrMsg());
            }
            ret = aclrtResetEvent(eventGroup.INPUT_E, stream_input);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(aclGetRecentErrMsg());
            }
            ret = aclrtResetEvent(eventGroup.COMPUTE_S, stream_compute);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(aclGetRecentErrMsg());
            }
            ret = aclrtResetEvent(eventGroup.COMPUTE_E, stream_compute);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(aclGetRecentErrMsg());
            }
            ret = aclrtResetEvent(eventGroup.OUTPUT_S, stream_output);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(aclGetRecentErrMsg());
            }
            ret = aclrtResetEvent(eventGroup.OUTPUT_E, stream_output);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(aclGetRecentErrMsg());
            }
            EventGroup new_task;
            std::swap(eventGroups[cur], new_task);
            Release();
        }
    }

    void Next() {
        cur = (cur + 1) % depth;
    }

    void Occupy() {
        active[cur] = 1;
    }

    void Release() {
        active[cur] = 0;
    }

    void SyncAndOccupy() {
        Sync();
        Occupy();
    }

    void SyncAll() {
        for (int i = 0; i < depth; i++) {
            Next();
            Sync();
        }
    }

    void Compute(ModelInferenceProcessor &modelInfer, void *inputDataSet, void *outputDataSet) {
        auto &eventGroup = CurrentEventGroup();
        // StreamWaitEvent(stream_compute, eventGroup.INPUT_E);
        StreamRecordEvent(stream_compute, eventGroup.COMPUTE_S);

        APP_ERROR ret = modelInfer.InferenceAsync(inputDataSet, outputDataSet,
                                                  stream_compute);
        if (ret != APP_ERR_OK) {
            throw std::runtime_error(GetError(ret));
        }

        StreamRecordEvent(stream_compute, eventGroup.COMPUTE_E);
    }

    void Host2Device(MirroredMemoryData &memData, bool skip = false) {
        auto &eventGroup = CurrentEventGroup();
        StreamRecordEvent(stream_input, eventGroup.INPUT_S);

        if (!skip) {
            for (int i = 0; i < memData.device_mem.size(); i++) {
                MemoryData &memDevice = memData.device_mem[i];
                MemoryData &memHost = memData.host_mem[i];
                APP_ERROR ret = MemoryHelper::MxbsMemcpyAsync(
                    memDevice, memHost, memDevice.size, stream_input);
                if (ret != APP_ERR_OK) {
                    ERROR_LOG("%s", aclGetRecentErrMsg());
                    throw std::runtime_error("Host2Device failed.");
                }
            }
        }

        StreamRecordEvent(stream_input, eventGroup.INPUT_E);
    }

    void Device2Host(MirroredMemoryData &memData, bool skip = false) {
        auto &eventGroup = CurrentEventGroup();
        // StreamWaitEvent(stream_output, eventGroup.COMPUTE_E);
        StreamRecordEvent(stream_output, eventGroup.OUTPUT_S);

        if (!skip) {
            for (int i = 0; i < memData.host_mem.size(); i++) {
                MemoryData &memDevice = memData.device_mem[i];
                MemoryData &memHost = memData.host_mem[i];
                APP_ERROR ret = MemoryHelper::MxbsMemcpyAsync(
                    memHost, memDevice, memHost.size, stream_output);
                if (ret != APP_ERR_OK) {
                    ERROR_LOG("%s", aclGetRecentErrMsg());
                    throw std::runtime_error("d2h failed.");
                }
            }
        }

        StreamRecordEvent(stream_output, eventGroup.OUTPUT_E);
    }

  private:
    int cur{}, depth;
    aclrtStream stream_input{}, stream_compute{}, stream_output{};
    std::vector<EventGroup> eventGroups;
    std::vector<int> active;

    EventGroup &CurrentEventGroup() { return eventGroups[cur]; }

    static void StreamRecordEvent(aclrtStream stream, aclrtEvent event) {
        aclError ret = aclrtRecordEvent(event, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
    }

    static void StreamWaitEvent(aclrtStream stream, aclrtEvent event) {
        aclError ret = aclrtStreamWaitEvent(stream, event);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(aclGetRecentErrMsg());
        }
    }
};
} // namespace Base

#endif // !_ASYNC_EXECUTOR_H
