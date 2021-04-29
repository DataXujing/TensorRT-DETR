
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


import torch
import torchvision
from torchsummary import summary
import time
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from trt_util.process_img import PyTorchTensorHolder
from trt_util.trt_lite import TrtLite

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


from model.hubconf import detr_resnet50
device = torch.device("cuda:0")


def time_static_torch(input_data,model_path='checkpoint/detr_resnet50.pth',batch_size=1,nRound=1000):
    detr = detr_resnet50(pretrained=False,num_classes=20+1).eval() 
    state_dict =  torch.load(model_path)   
    detr.load_state_dict(state_dict["model"])
    detr.to(device)

    torch.cuda.synchronize()
    t0 = time.time()

    for i in range(nRound):
        detr(input_data)

    torch.cuda.synchronize()

    Latency_pytorch = (time.time() - t0)*1000 / nRound
    Throughput_pytorch = 1000/Latency_pytorch*batch_size

    # 清空释放显存
    del detr
    input_data.cpu()
    del input_data
    torch.cuda.empty_cache()

    return Latency_pytorch, Throughput_pytorch



def time_static_trt(input_data,engine_path,batch_size=1,nRound=1000):
    trt_ = TrtLite(engine_file_path=engine_path)
    # trt_.print_info()
    # if batch_size == 1:
    #     i2shape = {0: (1, 3, 800, 800)}
    # else:
    #     i2shape = batch_size
    i2shape = {0: (batch_size, 3, 800, 800)}
    io_info = trt_.get_io_info(i2shape)
    d_buffers = trt_.allocate_io_buffers(i2shape, True)
    output_data_trt_prob = np.zeros(io_info[1][2], dtype=np.float32)
    output_data_trt_box = np.zeros(io_info[2][2], dtype=np.float32)

    d_buffers[0] = input_data

    torch.cuda.synchronize()
    t0 = time.time()

    for i in range(nRound):
        trt_.execute([t.data_ptr() for t in d_buffers], i2shape)
        # output_data_trt_prob = d_buffers[1].cpu().numpy()
        # output_data_trt_box = d_buffers[2].cpu().numpy()

    torch.cuda.synchronize()

    Latency_trt = (time.time() - t0) *1000  / nRound
    Throughput_trt = 1000/Latency_trt*batch_size

    # 释放显存
    del trt_
    input_data.cpu()
    del input_data
    d_buffers[0].cpu()
    try:
        del d_buffers[0]
        d_buffers[1].cpu()
        del d_buffers[1]
        d_buffers[2].cpu()
        del d_buffers[2]
    except:
        pass
    torch.cuda.empty_cache()

    return Latency_trt, Throughput_trt



if __name__ == "__main__":

    
    # # Latency and Throughput
    # Pytorch batch size = 32时 out of memory, 因此我们仅对比了batch size是16以下的batch
    # batch size =32 
    # RuntimeError: CUDA out of memory. 
    # Tried to allocate 314.00 MiB (GPU 0; 14.76 GiB total capacity; 13.03 GiB already allocated; 
    # 230.75 MiB free; 13.55 GiB reserved in total by PyTorch)

    batch_sizes = [16,8,4,2,1]
    # batch_sizes = [16]

    # FP32
    static_torch_32 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}
    static_trt_32 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}
    static_trt_16 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}

    for batch_size in batch_sizes:
        print(f"[INFO] 当前测试的batch size为：{batch_size}")
        torch.manual_seed(0)
        input_data = torch.randn(batch_size, 3, 800, 800, dtype=torch.float32, device='cuda')

        # torch
        print("[INFO] 正在进行pytorch测试")
        l_torch,t_torch = time_static_torch(input_data=input_data,model_path='checkpoint/detr_resnet50.pth',batch_size=batch_size,nRound=1000)

        print("[INFO] 释放模型")
        time.sleep(10)

        # fp32
        if batch_size == 1:
            batch_plan = "./detr.plan"
        else:
            batch_plan = f"./output/detr_batch_{batch_size}.plan"
        print("[INFO] 正在进行trt FP32测试")
        l_trt,t_trt = time_static_trt(input_data=input_data,engine_path=batch_plan,batch_size=batch_size,nRound=1000)

        lsu = round(l_trt / l_torch,2)
        tsu = round(t_trt /t_torch,2)

        time.sleep(10)

        # fp16
        if batch_size == 1:
            batch_plan = "./detr_fp16.plan"
        else:
            batch_plan = f"./output/detr_batch_{batch_size}_fp16.plan"
        print("[INFO] 正在进行trt FP16测试")
        l_trt_16,t_trt_16 = time_static_trt(input_data=input_data,engine_path=batch_plan,batch_size=batch_size,nRound=1000)

        lsu_16 = round(l_trt_16 / l_torch,2)
        tsu_16 = round(t_trt_16 /t_torch,2)

        input_data.cpu()
        del input_data
        torch.cuda.empty_cache()
        time.sleep(10)

        static_torch_32['batch_size'].append(batch_size)
        static_torch_32['latency'].append(l_torch)
        static_torch_32['throughput'].append(t_torch)
        static_torch_32['LSU'].append("1x")
        static_torch_32['TSU'].append("1x")

        static_trt_32['batch_size'].append(batch_size)
        static_trt_32['latency'].append(l_trt)
        static_trt_32['throughput'].append(t_trt)
        static_trt_32['LSU'].append(str(lsu)+"x")
        static_trt_32['TSU'].append(str(tsu)+"x")

        static_trt_16['batch_size'].append(batch_size)
        static_trt_16['latency'].append(l_trt_16)
        static_trt_16['throughput'].append(t_trt_16)
        static_trt_16['LSU'].append(str(lsu_16)+"x")
        static_trt_16['TSU'].append(str(tsu_16)+"x")

    print("-"*50)
    print("torch:")
    print(static_torch_32)
    print("trt fp32:")
    print(static_trt_32)
    print("trt fp16:")
    print(static_trt_16)
    print("-"*50)

    # plot latency vs throughput
    torch_x = static_torch_32['latency']
    torch_y = static_torch_32['throughput']

    trt_32_x = static_trt_32['latency']
    trt_32_y = static_trt_32['throughput']

    trt_16_x = static_trt_16['latency']
    trt_16_y = static_trt_16['throughput']

    plt.rcParams['figure.figsize'] = (16.0, 9.0)
    plt.plot(torch_x, torch_y, 'ro--',label='Pytorch')
    for i,(a, b) in enumerate(zip(torch_x, torch_y)):
        # plt.text(a+15,b-0.15,'(%d,%d,%d)'%(batch_size[i],a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'r'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'r'})


    plt.plot(trt_32_x, trt_32_y, 'b^--',label='TensorRT(FP32)')
    for i,(a, b) in enumerate(zip(trt_32_x, trt_32_y)):
        # plt.text(a+15,b-0.15,'(%d,%d)'%(a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'b'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'b'})


    plt.plot(trt_16_x, trt_16_y, 'g*--',label='TensorRT(FP16)')
    for i,(a, b) in enumerate(zip(trt_16_x, trt_16_y)):
        # plt.text(a+15,b-0.15,'(%d,%d)'%(a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'g'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'g'})
        if batch_sizes[i] in [4,8]:
            plt.annotate(f"({int(a)},{int(b)})",xy=(a,b),xytext=(a*0.9,b*0.9),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput')
    plt.legend()
    plt.savefig("./latency_vs_throughput.png")
    plt.close()



    

