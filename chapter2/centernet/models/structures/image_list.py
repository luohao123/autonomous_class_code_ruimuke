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
from __future__ import division

from typing import Any, List, Sequence, Tuple, Union

import torch
from torch.nn import functional as F


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w)
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w).
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]  # type: ignore

    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: Sequence[torch.Tensor],
        size_divisibility: int = 0,
        pad_ref_long: bool = False,
        pad_value: float = 0.0,
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded with `pad_value`
                so that they will have the same shape.
            size_divisibility (int): If `size_divisibility > 0`, also adds padding to ensure
                the common height and width is divisible by `size_divisibility`
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
        max_size = list(max(s) for s in zip(*[img.shape for img in tensors]))
        if pad_ref_long:
            max_size_max = max(max_size[-2:])
            max_size[-2:] = [max_size_max] * 2
        max_size = tuple(max_size)

        if size_divisibility > 0:
            import math

            stride = size_divisibility
            max_size = list(max_size)  # type: ignore
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)  # type: ignore
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)  # type: ignore
            max_size = tuple(max_size)

        image_sizes = [im.shape[-2:] for im in tensors]

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padded = F.pad(
                tensors[0],
                [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]],
                value=pad_value,
            )
            batched_imgs = padded.unsqueeze_(0)
        else:
            batch_shape = (len(tensors),) + max_size
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)