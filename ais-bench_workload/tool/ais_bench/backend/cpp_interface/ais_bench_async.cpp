#include <cstdio>
#include <iostream>
#include <iomanip>
#include <random>
#include <memory>

#include <thread>
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>

#include "Base/Tensor/TensorBuffer/TensorBuffer.h"
#include "Base/Tensor/TensorShape/TensorShape.h"
#include "Base/Tensor/TensorContext/TensorContext.h"
#include "Base/Tensor/TensorBase/TensorBase.h"

#include "Base/ModelInfer/SessionOptions.h"
#include "ModelInferenceProcessor.h"
#include "PyInferenceSession/PyInferenceSession.h"

int create_pure_input_tensors_bt(std::vector<Base::TensorDesc> descs, int deviceId, std::vector<Base::BaseTensor>& intensors)
{
    for (const auto& desc : descs) {
        auto buf = malloc(desc.realsize);
        Base::BaseTensor tensor{buf, desc.realsize};
        intensors.push_back(std::move(tensor));
    }
    return 0;
}

int str2num(char* str)
{
    int n = 0;
    int flag = 0;
    while (*str >= '0' && *str <= '9') {
        n = n * 10 + (*str - '0');
        str++;
    }
    if (flag == 1) {
        n = -n;
    }
    return n;
}

int main(int argc, char **argv) {
    std::string modelPath = argv[1];
    if (argc < 4) {
        return 0;
    }
    int loop = str2num(argv[2]);
    int threads = str2num(argv[3]);

    std::shared_ptr<Base::SessionOptions> options = std::make_shared<Base::SessionOptions>();
    options->loop = loop;
    options->log_level = 1;

    int deviceId = 0;
    std::shared_ptr<Base::PyInferenceSession> session = std::make_shared<Base::PyInferenceSession>(modelPath, deviceId, options);
    std::vector<Base::TensorDesc> indescs = session->GetInputs();
    std::vector<Base::BaseTensor> intensors = {};

    create_pure_input_tensors_bt(indescs, deviceId, intensors);
    Base::PerfOption perf_opt;
    perf_opt.loop = loop;
    perf_opt.threads = threads;

    auto elapsed_ns = Perf(modelPath, intensors, perf_opt);
    std::cout << "elapsed: " << std::setprecision(3) << elapsed_ns / 1000000.0 << "ms" << std::endl;
    return 0;
}
