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
import sys
import time
import cv2
from PIL import Image
import argparse

import pycuda.driver as cuda 
import pycuda.autoinit
import cupy as cp
import numpy as np
import tensorrt as trt


from trt_util.common import allocate_buffers,do_inference_v2,build_engine_onnx
from trt_util.process_img import preprocess_np,preprocess_torch_v1
from trt_util.plot_box import plot_box, CLASSES

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

   
def main(onnx_model_file,engine_file,fp16=False,batch_size=1):

    # Build a TensorRT engine.
    with build_engine_onnx(onnx_model_file,engine_file,FP16=fp16,batch_size=batch_size,verbose=False) as engine:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            print("------Engine Infor:---------")
            print(engine.max_batch_size)
            print(engine.get_binding_shape(0))
            print(engine.get_binding_shape(1))
            print(engine.get_binding_shape(2))

            print("------Context Infor:---------")
            print(context.get_binding_shape(0))
            print(context.get_binding_shape(1))
            print(context.get_binding_shape(1))
                   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create TensorRT Engine in FP32 ,FP16 Mode ')
    parser.add_argument('--model_dir', type= str , default='./output/detr_sim.onnx', help='ONNX Model Path')    
    parser.add_argument('--engine_dir', type= str , default='./output/detr_batch_.plan', help='TensorRT Engine File')

    parser.add_argument('--fp16', action="store_true", help='Open FP16 Mode or Not, if True You Should Load FP16 Engine File')
    parser.add_argument('--batch_size', type=int , default=2, help='Batch size, static=2')


    args = parser.parse_args()

    main(args.model_dir,args.engine_dir,args.fp16,args.batch_size)
