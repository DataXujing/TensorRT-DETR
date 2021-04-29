
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



def engine_infer(engine,context,inputs, outputs, bindings, stream,test_image): 

    # image_input, img_raw, _ = preprocess_np(test_image)
    image_input, img_raw, _ = preprocess_torch_v1((test_image))
    inputs[0].host = image_input.astype(np.float32).ravel()  # device to host to device,在性能对比时将替换该方式

    start = time.time()
    scores,boxs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, input_tensor=image_input)
    print(f"推断耗时：{time.time()-start}s")

    # print(scores)
    # print(boxs)

    output_shapes =  [(1,100,22), (1,100,4)]
    scores = scores.reshape(output_shapes[0])
    boxs = boxs.reshape(output_shapes[1])

    return scores,boxs,img_raw
  

    
def main(onnx_model_file,engine_file,image_dir,fp16=False,int8=False,batch_size=1,dynamic=False):
  
    test_images = [test_image for test_image in os.listdir(image_dir)]

    if int8:
        # only load the plan engine file
        if not os.path.exists(engine_file):
            raise "[Error] INT8 Mode must given the correct engine plan file. Please Check!!!"
        with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            # print(dir(context))

            if dynamic:
                context.active_optimization_profile = 0#增加部分
                origin_inputshape=context.get_binding_shape(0)
                if origin_inputshape[0]==-1:
                    origin_inputshape[0] = batch_size
                    context.set_binding_shape(0,(origin_inputshape))
            print(f"[INFO] INT8 mode.Dynamic:{dynamic}. Deserialize from: {engine_file}.")

            for test_image in test_images:

                scores,boxs, img_raw = engine_infer(engine,context,inputs, outputs, bindings, stream, os.path.join(image_dir,test_image))

                print(f"[INFO] trt inference done. save result in : ./trt_infer_res/in8/{test_image}")
                if not os.path.exists("./trt_infer_res/in8"):
                    os.makedirs("./trt_infer_res/in8")
                plot_box(img_raw, scores, boxs, prob_threshold=0.7, save_fig=os.path.join('./trt_infer_res/in8',test_image))
            

    else:
        # Build a TensorRT engine.
        with build_engine_onnx(onnx_model_file,engine_file,FP16=fp16,verbose=False,dynamic_input=dynamic) as engine:
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            # Contexts are used to perform inference.
            with engine.create_execution_context() as context:
                print(engine.get_binding_shape(0))
                print(engine.get_binding_shape(1))
                print(engine.get_binding_shape(2))

                print(context.get_binding_shape(0))
                print(context.get_binding_shape(1))
                # Load a normalized test case into the host input page-locked buffer.
                if dynamic:
                    context.active_optimization_profile = 0#增加部分
                    origin_inputshape=context.get_binding_shape(0)
                    if origin_inputshape[0]==-1:
                        origin_inputshape[0] = batch_size
                        context.set_binding_shape(0,(origin_inputshape))

                print(f"[INFO] FP16 mode is: {fp16},Dynamic:{dynamic} Deserialize from: {engine_file}.")

                for test_image in test_images:
                    scores,boxs, img_raw = engine_infer(engine,context,inputs, outputs, bindings, stream, os.path.join(image_dir,test_image))

                    if fp16:
                        save_dir = "./trt_infer_res/fp16"
                    else:
                        save_dir = "./trt_infer_res/fp32"

                    print(f"[INFO] trt inference done. save result in : {save_dir}/{test_image}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plot_box(img_raw, scores, boxs, prob_threshold=0.7, save_fig=os.path.join(save_dir,test_image))

                    
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference by TensorRT in FP32 ,FP16 Mode or INT8 Mode.')
    parser.add_argument('--model_dir', type= str , default='./detr_sim.onnx', help='ONNX Model Path')    
    parser.add_argument('--engine_dir', type= str , default='./detr.plan', help='TensorRT Engine File')
    parser.add_argument('--image_dir', type=str,default="./test", help='Test Image Dir')

    parser.add_argument('--fp16', action="store_true", help='Open FP16 Mode or Not, if True You Should Load FP16 Engine File')
    parser.add_argument('--int8', action="store_true", help='Open INT8 Mode or Not, if True You Should Load INT8 Engine File')
    parser.add_argument('--batch_size', type=int , default=1, help='Batch size, static=1')
    parser.add_argument('--dynamic', action="store_true", help='Dynamic Shape or Not when inference in trt')


    args = parser.parse_args()

    main(args.model_dir,args.engine_dir,args.image_dir,args.fp16,args.int8,args.batch_size,args.dynamic)


