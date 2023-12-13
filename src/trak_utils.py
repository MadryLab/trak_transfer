import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch as ch
from numpy.lib.format import open_memmap
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from trak_fair import TRAKer
from trak_fair.gradient_computers import (AbstractGradientComputer,
                                          FunctionalGradientComputer,
                                          IterativeGradientComputer)
from trak_fair.modelout_functions import (TASK_TO_MODELOUT,
                                          AbstractModelOutput,
                                          ImageClassificationModelOutput)
from trak_fair.projectors import AbstractProjector
from trak_fair.savers import AbstractSaver, MmapSaver
from trak_fair.score_computers import AbstractScoreComputer, BasicScoreComputer
from trak_fair.utils import parameters_to_vector

ch = torch
from trak_fair.utils import get_num_params, parameters_to_vector, vectorize


def open_json(result_path, name):
    with open(os.path.join(result_path, name), 'r') as f:
        metadata = json.load(f)
    return metadata


def get_scores(result_path, exp_name, num_targets=None):
    metadata = open_json(result_path, "metadata.json")
    train_set_size = metadata['train set size']
    if num_targets is None:
        exp_data = open_json(result_path, "experiments.json")
        num_targets = exp_data[exp_name]['num_targets']
    score_f = open_memmap(filename=os.path.join(result_path, "scores", f"{exp_name}.mmap"),
                          mode='r', shape=(train_set_size, num_targets), dtype=np.float16)
    return score_f


class VectorizedTransferGradientComputer(FunctionalGradientComputer):
    def __init__(self,
                 model: torch.nn.Module,
                 task: AbstractModelOutput,
                 grad_dim: int) -> None:
        super().__init__(model, task, grad_dim)
        self.model = model
        self.num_params = get_num_params(self.model.resnet)
        self.load_model_params(model)

    def load_model_params(self, model) -> None:
        """ Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())
        self.grad_func_weights = {k: v for k, v in model.named_parameters() if k.startswith('resnet')}
        self.class_func_weights = {k: v for k, v in model.named_parameters() if not k.startswith('resnet')}

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(self.modelout_fn.get_custom_output, has_aux=False, argnums=1)
        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = torch.empty(size=(batch[0].shape[0], self.num_params),
                            dtype=batch[0].dtype,
                            device=batch[0].device)
        vectorize(torch.func.vmap(grads_loss,
                                  in_dims=(None, None, None, None, *([0] * len(batch))),
                                  randomness='different')(self.model,
                                                          self.grad_func_weights,
                                                          self.class_func_weights,
                                                          self.func_buffers,
                                                          *batch), grads)
        return grads

class HFImageClassificationModelOutput(ImageClassificationModelOutput):

    @staticmethod
    def get_output(model: Module,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor) -> Tensor:
        logits = ch.func.functional_call(model,
                                         (weights, buffers),
                                         image.unsqueeze(0),
                                         kwargs={'return_dict': False})
        bindex = ch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        # remove the logits of the correct labels from the sum
        # in logsumexp by setting to -ch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = ch.tensor(-ch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()
    
    @staticmethod
    def get_custom_output(model: Module,
                   resnet_weights: Iterable[Tensor],
                   class_weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor) -> Tensor:
        weights = {**resnet_weights, **class_weights}
        logits = ch.func.functional_call(model,
                                         (weights, buffers),
                                         image.unsqueeze(0),
                                         kwargs={'return_dict': False}) #[0]
                                         
        bindex = ch.arange(logits.shape[0]).to(logits.device, non_blocking=False)

        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        # remove the logits of the correct labels from the sum
        # in logsumexp by setting to -ch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = ch.tensor(-ch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

    def get_out_to_loss_grad(self, model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        """ Computes the (reweighting term Q in the paper)

        Args:
            model (torch.nn.Module):
                torch model
            weights (Iterable[Tensor]):
                functorch model weights
            buffers (Iterable[Tensor]):
                functorch model buffers
            batch (Iterable[Tensor]):
                input batch

        Returns:
            Tensor:
                out-to-loss (reweighting term) for the input batch
        """
        images, labels = batch
        logits = ch.func.functional_call(model, (weights, buffers), images, kwargs={'return_dict': False})
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach().unsqueeze(-1)


class MemoryHogScoreComputer(BasicScoreComputer):
    def get_scores(self, features: Tensor, target_grads: Tensor) -> Tensor:
        train_dim = features.shape[0]
        target_dim = target_grads.shape[0]

        if target_dim < self.CUDA_MAX_DIM_SIZE:
            return features @ target_grads.T

        results = []
        #result = ch.empty(train_dim, target_dim, dtype=self.dtype, device='cpu')
        blocks = ch.split(target_grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        for i, block in tqdm(enumerate(blocks)):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(target_grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            results.append((features @ block.T).cpu())
            #result[:, start: end] = (features @ block.T).cpu()
        result = torch.cat(results, dim=1)
        print(result.shape)
        return result

class MemoryHogTRAKer(TRAKer):

    def finalize_scores(self,
                        exp_name: str,
                        model_ids: Iterable[int] = None,
                        allow_skip: bool = False,
                        skip_normalize: bool = False,
                        target_mask=None,
                        out_path=None) -> Tensor:
       
        # reset counter for inds used for scoring
        self._last_ind_target = 0

        if model_ids is None:
            model_ids = self.saver.model_ids
            finalized_model_ids = []
            for model_id in model_ids:
                json_path = os.path.join(self.saver.save_dir, f"id_{model_id}.json")
                valid = False
                if os.path.exists(json_path): 
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                        if metadata[str(model_id)]['is_finalized'] == 1:
                            valid = True
                if valid:
                    finalized_model_ids.append(model_id)
                else:
                    print("skipping", model_id)
            model_ids = finalized_model_ids
        else:
            model_ids = {model_id: self.saver.model_ids[model_id] for model_id in model_ids}
        assert len(model_ids) > 0, 'No model IDs to finalize scores for'

        if self.saver.experiments.get(exp_name) is None:
            raise ValueError(f'Experiment {exp_name} does not exist. Create it\n\
                              and compute scores first before finalizing.')
        if target_mask is None:
            target_mask = np.ones(self.saver.experiments[exp_name]['num_targets']) > 0
        if out_path is None:
            out_path = self.saver.save_dir.joinpath(f'scores/all_classes_scores.npy')
        #num_targets = self.saver.experiments[exp_name]['num_targets']
        num_targets = int(target_mask.sum())
        _completed = [False] * len(model_ids)


        _avg_out_to_losses = np.zeros((self.saver.train_set_size, 1),
                                      dtype=np.float16 if self.dtype == ch.float16 else np.float32)
        _scores_ram = torch.zeros((self.saver.train_set_size, num_targets), dtype=self.dtype, device=self.device)

        for j, model_id in enumerate(tqdm(model_ids, desc='Finalizing scores for all model IDs..')):
            self.saver.load_current_store(model_id)
            try:
                self.saver.load_current_store(model_id, exp_name, num_targets)
            except OSError as e:
                if allow_skip:
                    self.logger.warning(f'Could not read target gradients for model ID {model_id}. Skipping.')
                    continue
                else:
                    raise e

            if self.saver.model_ids[self.saver.current_model_id]['is_finalized'] == 0:
                self.logger.warning(f'Model ID {self.saver.current_model_id} not finalized, cannot score')
                continue

            g = ch.as_tensor(self.saver.current_store['features'], device=self.device)
            g_target = ch.as_tensor(self.saver.current_store[f'{exp_name}_grads'])[target_mask].to(self.device)

            _scores_ram += self.score_computer.get_scores(g, g_target)

            _avg_out_to_losses += self.saver.current_store['out_to_loss']
            _completed[j] = True

        _num_models_used = float(sum(_completed))
        _scores_ram = _scores_ram.cpu().numpy()
        if skip_normalize:
            _scores_ram = (_scores_ram / _num_models_used)
        else:
            _scores_ram = (_scores_ram / _num_models_used) * (_avg_out_to_losses / _num_models_used)
        np.save(out_path, _scores_ram)
        return _scores_ram

    
class CustomTRAKer(TRAKer):
    """ The main front-facing class for TRAK. See the `README
    <https://github.com/MadryLab/trak>`_ and `docs
    <https://trak.readthedocs.io/en/latest/>`_ for example usage.

    """
    def __init__(self,
                 model: torch.nn.Module,
                 task: Union[AbstractModelOutput, str],
                 train_set_size: int,
                 save_dir: str = './trak_results',
                 load_from_save_dir: bool = True,
                 device: Union[str, torch.device] = 'cuda',
                 gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
                 projector: Optional[AbstractProjector] = None,
                 saver: Optional[AbstractSaver] = None,
                 score_computer: Optional[AbstractScoreComputer] = None,
                 proj_dim: int = 2048,
                 logging_level=logging.INFO,
                 use_half_precision: bool = True,
                 proj_max_batch_size: int = 32,
                 grad_model: Optional[torch.nn.Module] = None,
                 ) -> None:
        self.model = model
        self.task = task
        self.train_set_size = train_set_size
        self.device = device
        self.dtype = ch.float16 if use_half_precision else ch.float32

        logging.basicConfig()
        self.logger = logging.getLogger('TRAK')
        self.logger.setLevel(logging_level)
        self.logger.warning('TRAK is still in an early 0.x.x version.\n\
                             Report any issues at https://github.com/MadryLab/trak/issues')
        if grad_model is None:
            self.num_params = get_num_params(self.model)
        else:
            self.num_params = get_num_params(grad_model)
        # inits self.projector
        self.init_projector(projector, proj_dim, proj_max_batch_size)

        # normalize to make X^TX numerically stable
        # doing this instead of normalizing the projector matrix
        self.normalize_factor = ch.sqrt(ch.tensor(self.num_params, dtype=ch.float32))

        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir

        if type(self.task) is str:
            self.task = TASK_TO_MODELOUT[self.task]()

        self.gradient_computer = gradient_computer(model=self.model,
                                                   task=self.task,
                                                   grad_dim=self.num_params)

        if score_computer is None:
            score_computer = BasicScoreComputer
        self.score_computer = score_computer(dtype=self.dtype,
                                             device=self.device)

        metadata = {
            'JL dimension': self.proj_dim,
            'JL matrix type': self.projector.proj_type,
            'train set size': self.train_set_size,
        }

        if saver is None:
            saver = MmapSaver
        self.saver = saver(save_dir=self.save_dir,
                           metadata=metadata,
                           train_set_size=self.train_set_size,
                           proj_dim=self.proj_dim,
                           load_from_save_dir=self.load_from_save_dir,
                           logging_level=logging_level,
                           use_half_precision=use_half_precision)

        self.ckpt_loaded = 'no ckpt loaded'
