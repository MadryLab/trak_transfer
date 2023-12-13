import os
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from tqdm import tqdm
from trak_fair import TRAKer

from src.loaders import get_loader
from src.models import CustomResNet
from src.trak_utils import HFImageClassificationModelOutput, MemoryHogScoreComputer, MemoryHogTRAKer
from src.utils import _get_inds, read_yaml
from torch.cuda.amp import autocast
# python run_trak.py --arch microsoft/resnet-18 --pretrained_config configs/imagenet.yaml --finetune_config configs/cifar10_transfer.yaml --model_directory imagenet_cifar_models/ --num_models 1 --traker_output test_results --set_grad_computer 1

@dataclass
class TrainingArguments:
    arch: str = field(default="facebook/opt-125m")
    pretrained_config: str = field(default='configs/imagenet.json')

@dataclass
class TrakArguments:
    model_directory: str = field(metadata={"help": "Path to models."})
    num_models: int = field(default=-1)
    traker_output: str = field(default="traker_results")
    model_id: int = field(default=-1)
    featurize: int = field(default=1)
    score: int = field(default=1)
    finalize_scores: int = field(default=1)
    skip_normalize: int = field(default=0)
    exp_name: str = field(default="downstream")
    finalize_class: int = field(default=-1)



def get_ds_loaders(c_args):
    train_inds = _get_inds('train', c_args)
    if train_inds is None:
        train_inds = np.arange(c_args['train_ds_size'])
    print("training set size", len(train_inds))
    common_args = {
        #'batch_size': c_args['batch_size'], 
        'batch_size': 200,
        'num_workers': c_args['num_workers'],
        'train_decoder_type': c_args['train_decoder_type']
    }
    return {
        'train': get_loader(
            path=c_args['train_path'], indices=train_inds, train_mode=False,
            **common_args
        ),
        'val': get_loader(
            path=c_args['val_path'], indices=_get_inds('val', c_args), train_mode=False,
            **common_args
        ),
        'test': get_loader(
            path=c_args['test_path'], indices=_get_inds('test', c_args), train_mode=False,
            **common_args
        ),
    }


def perform_scoring(traker, all_ckpts, model_ids, model_directory, exp_name, loader_targets, val_ds_size, model_creation_args):
    for i, ckpt_name in enumerate(tqdm(all_ckpts)):
        model_id = model_ids[i]
        ckpt_path = os.path.join(model_directory, ckpt_name, "checkpoint_last", "pytorch_model.bin")
        ckpt_model = fix_checkpoint(ckpt_path)
        traker.start_scoring_checkpoint(exp_name=exp_name,
                                        checkpoint=ckpt_model,
                                        model_id=model_id,
                                        num_targets=val_ds_size)
        for batch in tqdm(loader_targets):
            traker.score(batch=batch, num_samples=batch[0].shape[0])

def fix_checkpoint(sd_path):
    sd = torch.load(sd_path)
    return {k: v for k, v in sd.items() if 'secondary_classifier' not in k}


if __name__ == '__main__':
    parser = transformers.HfArgumentParser((TrainingArguments, TrakArguments))
    training_args, trak_args = parser.parse_args_into_dataclasses()
    pretrain_args = read_yaml(training_args.pretrained_config) 
    src_loaders = get_ds_loaders(pretrain_args)
    model_creation_args = {'arch': training_args.arch, 'num_src_labels': pretrain_args['num_classes'], 'num_dst_labels': -1}

    model_directory = trak_args.model_directory
    all_ckpts = sorted(os.listdir(model_directory))
    all_ckpts = [u for u in all_ckpts if os.path.exists(os.path.join(model_directory, u, "checkpoint_last", "pytorch_model.bin"))]
    if trak_args.model_id == -1:
        if trak_args.num_models != -1:
            all_ckpts = all_ckpts[:trak_args.num_models]
        model_ids = np.arange(len(all_ckpts))
    else:
        assert trak_args.model_id < len(all_ckpts)
        all_ckpts = [all_ckpts[trak_args.model_id]]
        model_ids = [trak_args.model_id]
    print(all_ckpts)
    assert len(all_ckpts) > 0

    # load the model
    model = CustomResNet(config=None, **model_creation_args)
    ckpt_path = os.path.join(model_directory, all_ckpts[0], "checkpoint_last", "pytorch_model.bin")
    model.load_state_dict(fix_checkpoint(ckpt_path))

    model = model.cuda().eval().half()
    model.set_grad_mode(do_overall_model=True, do_classifier=True, do_sec_classifier=False)
    
    # pretrain, only gradient on the backbone
    model.do_secondary = False

    traker_cls = MemoryHogTRAKer # TRAKer
    traker = traker_cls(model=model,
                    task=HFImageClassificationModelOutput(),
                    train_set_size=pretrain_args['train_ds_size'],
                    save_dir=trak_args.traker_output,
                    # score_computer=get_score_computer,
                   )

    if trak_args.featurize == 1:
        # scoring
        for i, ckpt_name in enumerate(tqdm(all_ckpts)):
            model_id = model_ids[i]
            ckpt_path = os.path.join(model_directory, ckpt_name, "checkpoint_last", "pytorch_model.bin")
            ckpt_model = fix_checkpoint(ckpt_path)
            traker.load_checkpoint(ckpt_model, model_id=model_id)

            for batch in tqdm(src_loaders['train']):
                # TRAKer computes features corresponding to the batch of examples,
                # using the checkpoint loaded above.
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        traker.finalize_features(model_ids=model_ids)

    # Tells TRAKer that we've given it all the information, at which point
    # TRAKer does some post-processing to get ready for the next step
    # (scoring target examples).


    if trak_args.score == 1:
        perform_scoring(
            traker=traker,
            all_ckpts=all_ckpts, 
            model_ids=model_ids,
            model_directory=model_directory,
            exp_name=trak_args.exp_name,
            loader_targets=src_loaders['val'],
            val_ds_size=pretrain_args['val_ds_size'],
            model_creation_args=model_creation_args,
        )

    if trak_args.finalize_scores == 1:
        assert trak_args.finalize_class != -1
        target_mask = np.load("val_in_labels.npy") == trak_args.finalize_class
        out_path = traker.saver.save_dir.joinpath(f'scores/{trak_args.finalize_class}_scores.npy')
        scores = traker.finalize_scores(exp_name=trak_args.exp_name, 
                                        skip_normalize=trak_args.skip_normalize,
                                        target_mask=target_mask,
                                        out_path=out_path,
                                        )
