#
# Copyright (c) 2020 jintian.
#
# This file is part of CenterNet_Pro_Max
# (see jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
from configs.tl.ct_tl_r50_config import config
from models.data import MetadataCatalog
from models.centernet import build_model
from models.train.checkpoint import DetectionCheckpointer
from models.data import transforms as T
import torch
from PIL import Image
import cv2
import sys
import os
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.get_dataset_label_map import coco_label_map_list

from alfred.dl.torch.common import device
import glob
import numpy as np


class ONNXExporter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        print('try load weights from: {}'.format(cfg.MODEL.WEIGHTS))
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, original_image):
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = np.array(original_image, dtype=np.float32)
    
        print('try exporting onnx model...')
        self.cfg.MODEL.ONNX = True
        self.cfg.MODEL.ONNX_POSTPROCESS = False
        onnx_model_f = 'centernet_r50_tl.onnx'
        aligned_img, _ = self.model.preprocess_onnx([image], torch.tensor([[height, width]], dtype=torch.int64),
                                                    target_shape=(512, 512))
        inp_dict = {
            'aligned_img': aligned_img,
        }
        print('onnx input image shape: {}'.format(aligned_img.shape))
        # 2 inputs how to trace model?
        torch.onnx.export(self.model, inp_dict, onnx_model_f, verbose=False)
        print('onnx exported!')


if __name__ == '__main__':
    # config.MODEL.WEIGHTS = 'weights/ct_r50_coco_model_0609999.pth'
    config.MODEL.WEIGHTS = 'weights/tl_r50_model_0274999.pth'
    # config.MODEL.WEIGHTS = 'checkpoints/ctdet_r50_coco_399999.pth'
    # config.MODEL.WEIGHTS = 'checkpoints/resnet50_centernet.pth'
    predictor = ONNXExporter(config)
    coco_label_map_list = ["trafficlight_red",
                           "trafficlight_green",
                           "trafficlight_black",
                           "trafficlight_yellow", ]

    if len(sys.argv) > 1:
        data_f = sys.argv[1]
    else:
        data_f = './images/tl.jpg'
    if os.path.isdir(data_f):
        img_files = glob.glob(os.path.join(data_f, '*.jpg'))
        for img_f in img_files:
            ori_img = cv2.imread(img_f)
            predictor(ori_img)
            exit(0)
    else:
        ori_img = cv2.imread(data_f)
        predictor(ori_img)
        
