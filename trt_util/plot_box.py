
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


import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cupy as cp


CLASSES = ["NA","Class A","Class B","Class C","Class D","Class E","Class F",
        "Class G","Class H","Class I","Class J","Class K","Class L","Class M",
        "Class N","Class O","Class P","Class Q","Class R","Class S","Class T"]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x = torch.from_numpy(x)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 将0-1映射到图像
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def plot_box(pil_img, prob, boxes, prob_threshold=0.1, save_fig=''):

    # 根据阈值将box去掉
    # print(prob)
    # print(boxes)
    prob = torch.from_numpy(prob).softmax(-1)[0,:,:-1]
    keep = prob.max(-1).values >= prob_threshold
    # convert boxes from [0; 1] to image scales
    prob =  prob.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()

    boxes = rescale_bboxes(boxes[0, keep], pil_img.size)
    prob = prob[keep]

    # print("----------------*--------------------")
    # print(f"prob: {prob}")
    # print(f"box: {boxes}")
    # print("----------------*--------------------")
    
    # plot box
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # plt.show()
    if not save_fig == '':
        plt.savefig(save_fig,transparent=True, dpi=300, pad_inches = 0)

    plt.close()