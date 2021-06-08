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
#    该部分代码参考了TensorRT官方示例完成，对相关方法进行修改
# 


import pycuda.driver as cuda  
#https://documen.tician.de/pycuda/driver.html
import pycuda.autoinit
import numpy as np
import tensorrt as trt
from .calibrator import Calibrator

import sys, os
import time

# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER = trt.Logger()

# Allocate host and device buffers, and create a stream.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))   # <--------- the main diff to v2
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream



def allocate_buffers_v2(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# do inference  multi outputs
def do_inference_v2(context, bindings, inputs, outputs, stream, input_tensor):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# The onnx path is used for Pytorch models. 
def build_engine_onnx(model_file,engine_file,FP16=False,verbose=False,dynamic_input=False,batch_size=1):

    def get_engine():
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network,builder.create_builder_config() as config, trt.OnnxParser(network,TRT_LOGGER) as parser:
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config,\
            trt.OnnxParser(network,TRT_LOGGER) as parser:
            # Workspace size is the maximum amount of memory available to the builder while building an engine.
            builder.max_workspace_size = 6 << 30 # 6G
            builder.max_batch_size = batch_size
            # config.max_batch_size = 2

            if FP16:
                print("[INFO] Open FP16 Mode!")
                # config.set_flag(tensorrt.BuilderFlag.FP16)
                builder.fp16_mode = True

            with open(model_file, 'rb') as model:
                parser.parse(model.read())
            if verbose:
                print(">"*50)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

            network.get_input(0).shape = [ batch_size, 3, 800, 800 ]

            if dynamic_input:
                profile = builder.create_optimization_profile();
                profile.set_shape("inputs", (1,3,800,800), (8,3,800,800), (64,3,800,800)) 
                config.add_optimization_profile(profile)

            # builder engine
            engine = builder.build_cuda_engine(network)
            print("[INFO] Completed creating Engine!")
            with open(engine_file, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file):
        # If a serialized engine exists, use it instead of building an engine.
        print("[INFO] Reading engine from file {}".format(engine_file))
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return get_engine()


# int8 quant
def build_engine_onnx_v2(onnx_file_path="", engine_file_path="",fp16_mode=False, int8_mode=False, \
               max_batch_size=1,calibration_stream=None, calibration_table_path="", save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network,\
                builder.create_builder_config() as config,trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit(f'[Error]ONNX file {onnx_file_path} not found')
            print(f'[INFO] Loading ONNX file from path {onnx_file_path}...')
            with open(onnx_file_path, 'rb') as model:
                print('[INFO] Beginning ONNX file parsing')
                parser.parse(model.read())
                assert network.num_layers > 0, '[Error] Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible '
            print('[INFO] Completed parsing of ONNX file')
            print(f'[INFO] Building an engine from file {onnx_file_path}; this may take a while...')        
            
            # build trt engine
            builder.max_batch_size = max_batch_size
            # config.max_workspace_size = 2 << 30 # 2GB
            builder.max_workspace_size = 2 << 30 # 2GB
            builder.fp16_mode = fp16_mode
            if int8_mode:
                builder.int8_mode = int8_mode
                # config.set_flag(trt.BuilderFlag.INT8)
                assert calibration_stream, '[Error] a calibration_stream should be provided for int8 mode'
                config.int8_calibrator  = Calibrator(calibration_stream, calibration_table_path)
                # builder.int8_calibrator  = Calibrator(calibration_stream, calibration_table_path)
                print('[INFO] Int8 mode enabled')
            engine = builder.build_cuda_engine(network) 
            if engine is None:
                print('[INFO] Failed to create the engine')
                return None   
            print("[INFO] Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print(f"[INFO] Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


