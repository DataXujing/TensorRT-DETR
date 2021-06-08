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

import io
import sys
import argparse

import numpy as np
import onnx
import onnxruntime
from onnxsim import simplify
import onnx_graphsurgeon as gs

import torch
from model.misc import nested_tensor_from_tensor_list
from model.hubconf import detr_resnet50


class ONNXExporter:

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, onnx_path,inputs_list, dynamic_axes=False, tolerate_small_mismatch=False, do_constant_folding=True,
                  output_names=None, input_names=None):
        model.eval()

        onnx_io = io.BytesIO()
        onnx_path = onnx_path

        torch.onnx.export(model, inputs_list[0], onnx_io,
            input_names=input_names, output_names=output_names,export_params=True,training=False,opset_version=12)
        torch.onnx.export(model, inputs_list[0], onnx_path,
            input_names=input_names, output_names=output_names,export_params=True,training=False,opset_version=12)
        
        print(f"[INFO] ONNX model export success! save path: {onnx_path}")

        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or isinstance(test_inputs, list):
                    # test_inputs = (nested_tensor_from_tensor_list(test_inputs),)
                    test_inputs = (test_inputs,)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

        print("[INFO] Validate the exported model with onnx runtime success!")

        # dynamic_shape
        if dynamic_axes:
            # dynamic_axes = [int(ax) for ax in list(dynamic_axes)]
            torch.onnx.export(model, inputs_list[0], './detr_dynamic.onnx', dynamic_axes={input_names[0]: {0:'-1'},output_names[0]:{0:'-1'},output_names[1]:{0:'-1'}}, 
                input_names=input_names, output_names=output_names, verbose=True, opset_version=12)
            
            print(f"[INFO] Dynamic Shape ONNX model export success! Dynamic shape:{dynamic_axes} save path: ./detr_dynamic.onnx")

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    @staticmethod
    def check_onnx(onnx_path):
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"[INFO]  ONNX model: {onnx_path} check success!")


    @staticmethod
    def onnx_change(onnx_path):
        '''该部分代码由导师提供，解决trt inference 全是0的问题，感谢!!!
        '''
        node_configs = [(2648,2684),(2775,2777),(2961,2963),(3333,3335),(4077,4079)]
        if 'batch_2' in onnx_path:
            node_number = node_configs[1]
        elif 'batch_4' in onnx_path:
            node_number = node_configs[2]
        elif 'batch_8' in onnx_path:
            node_number = node_configs[3]
        elif 'batch_16' in onnx_path:
            node_number = node_configs[4]
        else:
            node_number = node_configs[0]

        graph = gs.import_onnx(onnx.load(onnx_path))
        for node in graph.nodes:
            if node.name == f"Gather_{node_number[0]}":
                print(node.inputs[1])
                node.inputs[1].values = np.int64(5)
                print(node.inputs[1])
            elif node.name == f"Gather_{node_number[1]}":
                print(node.inputs[1])
                node.inputs[1].values = np.int64(5)
                print(node.inputs[1])
                
        onnx.save(gs.export_onnx(graph),onnx_path)
        print(f"[INFO] onnx修改完成, 保存在{onnx_path}.")





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DETR Model to ONNX Model')
    parser.add_argument('--model_dir', type= str , default='./checkpoint/detr_resnet50.pth', help='DETR Pytorch Model Saved Dir')    
    parser.add_argument('--dynamic_axes', action="store_true", help='Dynamic ONNX Model')
    parser.add_argument('--check',  action="store_true", help='Check Your ONNX Model')
    parser.add_argument('--onnx_dir', type=str,default="./detr.onnx", help="Check ONNX Model's dir")
    parser.add_argument('--batch_size', type=int,default=1, help="Batch Size")


    args = parser.parse_args()

    # load torch model
    detr = detr_resnet50(pretrained=False,num_classes=20+1).eval()  # max label index add 1
    state_dict =  torch.load(args.model_dir,map_location='cuda')     # pytorch model path
    detr.load_state_dict(state_dict["model"])
    
    # input
    dummy_image = [torch.ones(args.batch_size, 3, 800, 800) ]

    # to onnx
    onnx_export = ONNXExporter()
    onnx_export.run_model(detr,args.onnx_dir, dummy_image,input_names=['inputs'],dynamic_axes=args.dynamic_axes,
        output_names=["pred_logits", "pred_boxes"],tolerate_small_mismatch=True)

    # check onnx model
    if args.check:
        ONNXExporter.check_onnx(args.onnx_dir)


    print('[INFO] Simplifying model...')
    model = onnx.load(args.onnx_dir)
    # simplifying dynamic model
    simplified_model, check = simplify(model,
                                       input_shapes={'inputs': [args.batch_size, 3, 800, 800]},
                                       dynamic_input_shape=args.dynamic_axes)


    onnx.save(simplified_model,(args.onnx_dir[:-5]+"_sim.onnx"))

    # onnx change
    onnx_export.onnx_change(args.onnx_dir[:-5]+"_sim.onnx")


    # simplifer onnx 
    # $ python3 -m onnxsim detr.onnx detr_sim.onnx
    # $ python3 -m onnxsim detr_dynamic.onnx detr_dynamic_sim.onnx

    







    
