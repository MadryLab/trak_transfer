import pickle as pkl

import numpy as np
import torch

from src.ffcv_utils import get_ffcv_loader
import tqdm
import ffcv.fields.decoders as decoders
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate
from dataclasses import replace
from typing import Callable, Optional, Tuple
import torch.nn as nn

import numpy as np
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import AllocationQuery, Operation
from ffcv.pipeline.state import State
# These take in the desired image size, and the beton image size

IMAGE_DECODERS = {
    'simple': lambda imgsz: decoders.SimpleRGBImageDecoder(),
    'resized_crop': lambda imgsz: decoders.ResizedCropRGBImageDecoder((imgsz, imgsz)),
    'random_resized_crop': lambda imgsz: decoders.RandomResizedCropRGBImageDecoder((imgsz, imgsz)),
    'center_crop_256': lambda imgsz: decoders.CenterCropRGBImageDecoder((imgsz, imgsz), 224/256),
}





def get_loader(batch_size, num_workers, path, indices, train_mode, train_decoder_type, multiclass=-1):
    common_args = {
        'path': path, 'batch_size': batch_size, 'num_workers': num_workers, 'indices': indices,
        'label_idx': multiclass
    }
    if train_mode:
        spec_args = {
            'shuffle': True, 'drop_last': True, 'img_decoder': IMAGE_DECODERS[train_decoder_type](224),
            'custom_img_transform': [RandomHorizontalFlip()],
        }
    else:
        spec_args = {
            'shuffle': False, 'drop_last': False, 'img_decoder': IMAGE_DECODERS['center_crop_256'](224),
        }
    return get_ffcv_loader(**common_args, **spec_args)

