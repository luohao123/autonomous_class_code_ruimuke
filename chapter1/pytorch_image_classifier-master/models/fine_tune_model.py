# -*- coding: utf-8 -*-
# file: fine_tune_model.py
# author: JinTian
# time: 10/05/2017 9:54 AM
# Copyright 2017 JinTian. All Rights Reserved.
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
# ------------------------------------------------------------------------
from torchvision import models
from torch import nn
from alfred.dl.torch.common import device



class NewbieNet(nn.Module):

    def __init__(self, n_classes=10):
        super(NewbieNet, self).__init__()

        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3, 128, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 512, 3)
        self.relu2 = nn.ReLU()

        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(512, self.n_classes)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.f1(x)
        x = self.l1(x)
        print(x.shape)
        return x



def fine_tune_model():
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    # fine tune we change original fc layer into classes num of our own
    model_ft.fc = nn.Linear(num_features, 2)
    model_ft = model_ft.to(device)
    return model_ft
