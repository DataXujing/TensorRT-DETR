
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#                ~~~Medcare AI Lab~~~

import os
import torch
import torchvision
from torchsummary import summary
import time
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from trt_util.process_img import PyTorchTensorHolder
from trt_util.trt_lite import TrtLite
from trt_util.process_img import preprocess_torch_v1,preprocess_torch_v2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from model.hubconf import detr_resnet50
device = torch.device("cuda:0")

# model torch
model_path='checkpoint/detr_resnet50.pth'
detr = detr_resnet50(pretrained=False,num_classes=20+1).eval()  # <------这里类别需要+1
state_dict =  torch.load(model_path)   # <-----------修改加载模型的路径
detr.load_state_dict(state_dict["model"])
detr.to(device)

# model trt fp32
engine_path="./detr.plan"
trt_32 = TrtLite(engine_file_path=engine_path)

# model trt fp16
engine_path_16="./detr_fp16.plan"
trt_16 = TrtLite(engine_file_path=engine_path_16)


def acc_static_torch(test_dir,detr=detr):

    img, _, _ = preprocess_torch_v2(test_dir)
    outputs = detr(img.to(device))

    scores = outputs['pred_logits'].cpu().detach().numpy()
    boxes = outputs['pred_boxes'].cpu().detach().numpy()

    return scores.ravel(), boxes.ravel()


def acc_static_trt(test_dir,trt_=trt_32):
    
    # trt_.print_info()
    i2shape = {0: (1, 3, 800, 800)}
    io_info = trt_.get_io_info(i2shape)
    d_buffers = trt_.allocate_io_buffers(i2shape,True)
    output_data_trt_prob = np.zeros(io_info[1][2], dtype=np.float32)
    output_data_trt_box = np.zeros(io_info[2][2], dtype=np.float32)

    img, _, _ = preprocess_torch_v2(test_dir)
    d_buffers[0] = img.to(device)

    trt_.execute([t.data_ptr() for t in d_buffers], i2shape)

    scores = d_buffers[1].cpu().numpy().ravel()
    boxes = d_buffers[2].cpu().numpy().ravel()

    return scores, boxes




if __name__ == "__main__":
    
    # 平均相对精度的计算
    # np.mean(np.abs(output_data_pytorch - output_data_trt) / np.abs(output_data_pytorch))
    files = os.listdir("./test")

    Average_diff_perc_socre_32 = []
    Average_diff_perc_box_32 = []

    Average_diff_perc_socre_16 = []
    Average_diff_perc_box_16 = []

    for file in files[:1000]:
        print(file)
        file_path = os.path.join("./test/",file)

        torch_score, torch_box =  acc_static_torch(file_path)

        # fp32
        trt_score_32,trt_box_32 = acc_static_trt(file_path,trt_=trt_32)

        # fp16
        trt_score_16,trt_box_16 = acc_static_trt(file_path,trt_=trt_16)

        adp_score_32 = np.mean(np.abs(torch_score - trt_score_32) / np.abs(torch_score))
        adp_box_32 = np.mean(np.abs(torch_box - trt_box_32) / np.abs(torch_box))

        adp_score_16 = np.mean(np.abs(torch_score - trt_score_16) / np.abs(torch_score))
        adp_box_16 = np.mean(np.abs(torch_box - trt_box_16) / np.abs(torch_box))

        Average_diff_perc_socre_32.append(adp_score_32)
        Average_diff_perc_box_32.append(adp_box_32)

        Average_diff_perc_socre_16.append(adp_score_16)
        Average_diff_perc_box_16.append(adp_box_16)


    print("-"*50)
    print(f"trt FP32 Score的平均相对精度：{Average_diff_perc_socre_32}")
    print(f"trt FP32 Box的平均相对精度：{Average_diff_perc_box_32}")

    print(f"trt FP16 Score的平均相对精度：{Average_diff_perc_socre_16}")
    print(f"trt FP16 Box的平均相对精度：{Average_diff_perc_box_16}")
    print("-"*50)

    # plot Average diff percentage
    plt.rcParams['figure.figsize'] = (16.0, 9.0) # 单位是inches
    fig,subs=plt.subplots(2,2)
    subs[0][0].plot(np.arange(len(files[:1000])), Average_diff_perc_socre_32, 'ro',label='FP32(Score)')
    subs[0][0].axhline(y=1e-6,color='b',linestyle='--')
    subs[0][0].axhline(y=1e-5,color='b',linestyle='--')

    subs[0][0].set_xlabel('Test Image ID')
    subs[0][0].set_ylabel('Average Diff Percentage')
    subs[0][0].legend()

    subs[0][1].plot(np.arange(len(files[:1000])), Average_diff_perc_box_32, 'bv',label='FP32(Box)')
    subs[0][1].axhline(y=1e-6,color='r',linestyle='--')
    subs[0][1].axhline(y=1e-5,color='r',linestyle='--')
    subs[0][1].set_xlabel('Test Image ID')
    subs[0][1].set_ylabel('Average Diff Percentage')
    subs[0][1].legend()

    subs[1][0].plot(np.arange(len(files[:1000])), Average_diff_perc_socre_16, 'g^',label='FP16(Score)')
    subs[1][0].axhline(y=1e-3,color='k',linestyle='--')
    subs[1][0].axhline(y=1e-2,color='k',linestyle='--')
    subs[1][0].set_xlabel('Test Image ID')
    subs[1][0].set_ylabel('Average Diff Percentage')
    subs[1][0].legend()

    subs[1][1].plot(np.arange(len(files[:1000])), Average_diff_perc_box_16, 'k*',label='FP16(Box)')
    subs[1][1].axhline(y=1e-3,color='g',linestyle='--')
    subs[1][1].axhline(y=1e-2,color='g',linestyle='--')
    subs[1][1].set_xlabel('Test Image ID')
    subs[1][1].set_ylabel('Average Diff Percentage')
    subs[1][1].legend()

    plt.savefig("./average_diff_percentage.png")
    plt.close()






 

















