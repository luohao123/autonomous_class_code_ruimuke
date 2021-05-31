#!/usr/bin/python3
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

import torch.nn as nn
import torchvision.models.shufflenetv2 as shufflenetv2

from .backbone import Backbone
from alfred.utils.log import logger as logging

"""

Due to CenterNet only support 2048 out channel, only x2 support,
this is also the bestest version of shufflenetv2 described in paper.
however, x2 doesn't provide pretrained weights by torchvision, we gonna need
training from scratch

"""
_shufflenetsv2_mapper = {
    5: shufflenetv2.shufflenet_v2_x0_5,
    10: shufflenetv2.shufflenet_v2_x1_0,
    15: shufflenetv2.shufflenet_v2_x1_5,
    20: shufflenetv2.shufflenet_v2_x2_0,
}


class ShuffleNetV2Backbone(Backbone):

    def __init__(self, cfg, input_shape=None, pretrained=False):
        super().__init__()
        depth = cfg.MODEL.SHUFFLENETS.DEPTH
        backbone = _shufflenetsv2_mapper[depth](pretrained=pretrained)

        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.conv5 = backbone.conv5

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # logging.info('out shuffle: {}'.format(x.shape))
        # drop this maxpool due to CenterNet need 2048 feature channel
        # x = x.mean([2, 3])
        return x
