from dataclasses import replace
from typing import Callable, Optional, Tuple

import numpy as np
import torch as ch
import torch.nn as nn
from ffcv.fields.basics import IntDecoder
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
# from trainer import Trainer
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import AllocationQuery, Operation
from ffcv.pipeline.state import State
from ffcv.transforms import (Convert, RandomHorizontalFlip, Squeeze, ToDevice,
                             ToTensor)
# import antialiased_cnns
from ffcv.transforms.ops import ToTorchImage
from torchvision.transforms import Normalize

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class SelectLabel(Operation):
    """Select label from multiple labels of specified images.
    Parameters
    ----------
    indices : Sequence[int] / list
        The indices of labels to select.
    """

    def __init__(self, indices):
        super().__init__()

        assert isinstance(indices, list) or isinstance(indices, int), f"required dtype: int/list(int). received {type(indices)}"
        if isinstance(indices, int):
            indices = [indices]
        assert len(indices) > 0, "Number of labels to select must be > 0"
        self.indices = np.sort(indices)

    def generate_code(self) -> Callable:

        to_select = self.indices
        my_range = Compiler.get_iterator()

        def select_label(labels, temp_array, indices):
            new_shape = (labels.shape[0], len(to_select))
            labels_subset = np.zeros(shape=new_shape, dtype=labels.dtype)
            for i in my_range(labels.shape[0]):
                labels_subset[i] = labels[i][to_select]
            return labels_subset

        select_label.is_parallel = True
        select_label.with_indices = True

        return select_label

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        new_shape = (len(self.indices),)
        new_state = replace(previous_state, shape=new_shape)
        mem_allocation = AllocationQuery(new_shape, previous_state.dtype)
        # We do everything in place
        return (new_state, mem_allocation)


def get_ffcv_loader(path,
                    batch_size=1024,
                    num_workers=80,
                    indices=None,
                    img_decoder=SimpleRGBImageDecoder(),
                    custom_img_transform=[], # pre normalization, on cpu
                    shuffle=True,
                    drop_last=None,
                    pipeline_keys=['image', 'label'],
                    label_idx=-1,
                ):
    quasi_random = False # this might need to be true for supercloud?
    RANDOM_ORDER = OrderOption.QUASI_RANDOM if quasi_random else OrderOption.RANDOM
    order = RANDOM_ORDER if shuffle else OrderOption.SEQUENTIAL

    image_pipeline = [img_decoder,
                      *custom_img_transform,
                      ToTensor(),
                      ToDevice(ch.device('cuda'), non_blocking=True),
                      ToTorchImage(),
                      Convert(ch.float16),
                      Normalize((IMAGENET_MEAN * 255).tolist(), (IMAGENET_STD * 255).tolist())]
    if label_idx == -1:
        label_pipeline= [IntDecoder(),
                            ToTensor(),
                            Squeeze(),
                            ToDevice(ch.device('cuda'), non_blocking=True)]
    else:
        label_pipeline= [
            NDArrayDecoder(),
            SelectLabel(label_idx),
            ToTensor(),
            Squeeze(),
            Convert(ch.int64),
            ToDevice(ch.device('cuda'), non_blocking=True)
        ]

    pipelines = {'image': image_pipeline,'label': label_pipeline}
    pipelines = {k: v if k in pipeline_keys else None for k,v in pipelines.items()}
    return Loader(fname=path,
                  batch_size=batch_size,
                  num_workers=num_workers,
                  order=order,
                  os_cache=not quasi_random,
                  indices=indices,
                  pipelines=pipelines,
                  drop_last=drop_last)
