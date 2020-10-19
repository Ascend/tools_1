# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''

import torch
import torch.nn as nn


class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropy, self).__init__()
        self.on_value = 1.0 - smooth_factor
        self.off_value = 1.0 * smooth_factor / (num_classes - 1)

    def forward(self, input, target):
        one_hot_label = torch.npu_one_hot(target, -1, input.size(1), self.on_value, self.off_value)
        one_hot_label = one_hot_label.to(torch.float16)
        loss = torch.npu_softmax_cross_entropy_with_logits(input.to(torch.float16), one_hot_label)

        loss = torch.mean(loss, [0], keepdim=False, dtype=torch.float32)
        return loss

class LabelSmoothingNpu(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingNpu, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

        self.epsilon = 0.1
        self.num_classes = 1000

    def forward(self, x, target):
        CALCULATE_DEVICE = x.device
        logprobs = torch.nn.functional.log_softmax(x, dim=-1).to("cpu")

        targets = torch.zeros_like(logprobs).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * logprobs).mean(0).sum()

        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)
        # smooth_loss = -logprobs.mean(dim=-1)
        # loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.to(CALCULATE_DEVICE)


class LabelSmoothingGpu(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingGpu, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    #     print("----------------------LabelSooothing.__init__")
    # def __call__(self,x,target):
    #     print("----------------------LabelSooothing.__call__")
    #     return self.forward(self,x,target)

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        #print("================",type(x),x.size())
        #print("------------------",type(target),target.size(),target)
        return loss.mean()
