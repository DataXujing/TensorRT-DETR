
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
import glob
import cv2
from PIL import Image
import numpy as np
import argparse

import torchvision.transforms as T
from trt_util.common import build_engine_onnx_v2
from trt_util.calibrator import Calibrator


transform = T.Compose([
    T.Resize((800,800)),  # PIL.Image.BILINEAR
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(img_pil):
    img = transform(img_pil).cpu().numpy()
    return img

# def preprocess(img_pil):
#     img = img_pil.resize((800, 800),Image.BILINEAR)
#     img = np.array(img).astype(np.float32) / 255.0
#     img = img.transpose(2,0,1)
#     # print(img.shape)
#     img = (img - np.array([ [[0.485]], [[0.456]], [[0.406]] ]))/np.array([ [[0.229]], [[0.224]], [[0.225]] ])

#     # img = img.transpose(1,2,0)
#     # img = np.expand_dims(img, axis=0)
#     img = np.ascontiguousarray(img)
#     img = np.array(img).astype(np.float32)
#     print(img.shape)
#     return img

class DataLoader:
    def __init__(self,calib_img_dir="./calib_train_image",batch=1,batch_size=32):
        self.index = 0
        self.length = batch
        self.batch_size = batch_size
        self.calib_img_dir = calib_img_dir
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(self.calib_img_dir, "*.jpg"))
        print(f'[INFO] found all {len(self.img_list)} images to calib.')
        assert len(self.img_list) > self.batch_size * self.length, '[Error] {} must contains more than {} images to calib'.format(self.calib_img_dir,self.batch_size * self.length)
        self.calibration_data = np.zeros((self.batch_size,3,800,800), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), '[Error] Batch not found!!'
                # data preprocess
                img = Image.open(self.img_list[i + self.index * self.batch_size])
                # img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                # self.calibration_data[i] = np.ones((3,800,800), dtype=np.float32)
                self.calibration_data[i] = img

            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

def main(onnx_model_path,engine_model_path,calib_img_dir,calibration_table,fp16,int8,batch,batch_size):

    fp16_mode = fp16 
    int8_mode = int8 

    # calibration
    calibration_stream = DataLoader(calib_img_dir=calib_img_dir,batch=batch,batch_size=batch_size)
    engine_model_path = engine_model_path

    # 校准产生校准表，但是我们并没有生成校准表！
    engine_fixed = build_engine_onnx_v2(onnx_model_path, engine_model_path, fp16_mode=fp16_mode, 
        int8_mode=int8_mode,max_batch_size=batch_size, calibration_stream=calibration_stream, 
        calibration_table_path=calibration_table, save_engine=True)
    assert engine_fixed, '[Error] Broken engine_fixed'
    print('[INFO]  ====> onnx to tensorrt completed !\n')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TensorRT INT8 Quant.')
    parser.add_argument('--onnx_model_path', type= str , default='./detr_sim.onnx', help='ONNX Model Path')    
    parser.add_argument('--engine_model_path', type= str , default='./detr_int8.plan', help='TensorRT Engine File')
    parser.add_argument('--calib_img_dir', type= str , default='./calib_train_image', help='Calib Image Dir')   
    parser.add_argument('--calibration_table', type=str,default="./detr_calibration.cache", help='Calibration Table')
    parser.add_argument('--batch', type=int,default=958, help='Number of Batch: [total_image/batch_size]')  # 30660/batch_size
    parser.add_argument('--batch_size', type=int,default=32, help='Batch Size')

    parser.add_argument('--fp16', action="store_true", help='Open FP16 Mode')
    parser.add_argument('--int8', action="store_true", help='Open INT8 Mode')

    args = parser.parse_args()
    main(args.onnx_model_path,args.engine_model_path,args.calib_img_dir,args.calibration_table,
        args.fp16,args.int8,args.batch,args.batch_size)

    # python3 trt_int8_quant.py --onnx_model_path ./detr_sim.onnx --engine_model_path ./detr_int8.plan --calib_img_dir ./calib_train_image --calibration_table ./detr_calibration.cache --batch 1 --int8

