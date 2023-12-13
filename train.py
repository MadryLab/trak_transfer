# from wandb import Config
import copy
import os
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import tqdm
import transformers
import yaml
from torch.cuda.amp import autocast

from src.loaders import get_loader
from src.models import CustomResNet
from src.trainer import LightWeightTrainer
from src.utils import read_yaml, _get_inds
#python train.py --arch microsoft/resnet-18 --pretrained_config configs/imagenet.yaml --load_prev_pretrained /mnt/xfs/projects/trak_transfer/cfs/cf_results/in_cifar_top/0_0/checkpoint_last/pytorch_model.bin --finetune_config configs/cifar10_fullnet_transfer.yaml --output_dir test --exclude_file /mnt/xfs/projects/trak_transfer/cfs/cf_orders/in_cifar_top.npy --num_to_exclude 0 --save_checkpoint 0 --freeze_source 1
@dataclass
class TrainingArguments:
    arch: str = field(default="facebook/opt-125m")
    pretrained_config: str = field(default='configs/imagenet.json')
    finetune_config: str = field(default='configs/cifar10.json')
    load_prev_pretrained: str = field(default='')
    output_dir: str = field(default='')
    save_checkpoint: int = field(default=1)
    save_output: int = field(default=1)
    freeze_source: int = field(default=1)

@dataclass
class CounterfactualArguments:
    exclude_file: str = field(default='')
    num_to_exclude: int = field(default=0)
    class_mode: int = field(default=0)



def get_training_loaders(c_args, cf_args=None):
    train_inds = _get_inds('train', c_args)
    if train_inds is None:
        train_inds = np.arange(c_args['train_ds_size'])
    if cf_args is not None and cf_args.exclude_file != '':
        if cf_args.exclude_file == 'random':
            subset = np.arange(len(train_inds))
            np.random.shuffle(subset)
            subset = subset[cf_args.num_to_exclude:]
        else:
            subset = np.load(cf_args.exclude_file)
            subset = subset[cf_args.num_to_exclude:]
            if cf_args.class_mode == 1:
                print("class wise")
                in_labels = np.load("in_labels.npy")
                subset = np.isin(in_labels, subset)
        train_inds = train_inds[subset]
    print("training set size", len(train_inds))
    common_args = {
        'batch_size': c_args['batch_size'], 'num_workers': c_args['num_workers'],
        'train_decoder_type': c_args['train_decoder_type'],
        'multiclass': c_args.get('multiclass', -1)
    }
    return {
        'train': get_loader(
            path=c_args['train_path'], indices=train_inds, train_mode=True,
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

def get_training_args(c_args):
    return {
        'epochs': c_args['epochs'], 'lr': c_args['lr'],
        'weight_decay': c_args['weight_decay'], 'momentum': c_args['momentum'],
        'lr_scheduler': c_args['lr_scheduler'], 'step_size': c_args['step_size'],
        'lr_milestones': c_args['lr_milestones'], 'gamma': c_args['gamma'],
        'lr_peak_epoch': c_args['lr_peak_epoch'],
        'eval_epochs': c_args['eval_epochs']
    }

def evaluate_model(model, loader):
    criterion = nn.CrossEntropyLoss(reduction='none')
    logits_list = []
    label_list = []
    with torch.no_grad():
        with autocast():
            for img, labels in tqdm.tqdm(loader):
                out = model(img).cpu()
                logits_list.append(out)
                label_list.append(labels.cpu())
    all_logits = torch.cat(logits_list).cuda()
    all_labels = torch.cat(label_list).cuda()
    all_losses = criterion(all_logits, all_labels)
    all_preds = all_logits.argmax(-1)
    
    accuracy = (all_preds == all_labels).float().mean().item()
    print("------- EVAL ---------")
    with torch.no_grad():
        print("resnet sum", model.resnet.encoder.stages[0].layers[0].layer[0].convolution.weight.sum().item())
        print("cls sum", model.classifier[1].weight.sum().item())
        print("sec cls sum", model.secondary_classifier[1].weight.sum().item())
    print("accuracy", accuracy)
    return {'preds': all_preds.cpu().numpy(), 
            'labels': all_labels.cpu().numpy(), 
            'losses': all_losses.cpu().numpy(),
            'acc': accuracy}


def main(training_args, cf_args):
    pretrain_args = read_yaml(training_args.pretrained_config) 
    finetune_args = read_yaml(training_args.finetune_config) 
    model = CustomResNet(config=None, arch=training_args.arch, num_src_labels=pretrain_args['num_classes'], num_dst_labels=finetune_args['num_classes'])
    model = model.cuda().train()

    # Pretrain
    src_loaders = get_training_loaders(pretrain_args, cf_args)
    if training_args.load_prev_pretrained != '':
        print("loading from", training_args.load_prev_pretrained)
        ckpt_model = torch.load(training_args.load_prev_pretrained)
        new_state_dict = {k: v for k, v in ckpt_model.items() if not k.startswith('secondary_classifier')}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.set_grad_mode(do_overall_model=True, do_classifier=True, do_sec_classifier=False)
        model.do_secondary = False
        trainer = LightWeightTrainer(training_dir=training_args.output_dir, 
                                    save_intermediate=False,
                                    training_args=get_training_args(pretrain_args))
        trainer.fit(model, src_loaders['train'], src_loaders['val'])

   
    #model.set_grad_mode(do_overall_model=False, do_classifier=False, do_sec_classifier=True)
    model.do_secondary = True
    # finetune
    if training_args.freeze_source == 1:
        print("fixed feature")
        model = model.eval()
        model.set_grad_mode(do_overall_model=False, do_classifier=False, do_sec_classifier=True)
    else:
        print("full network")
        model.set_grad_mode(do_overall_model=True, do_classifier=False, do_sec_classifier=True)
    

    dst_loaders = get_training_loaders(finetune_args)
    trainer = LightWeightTrainer(training_dir=training_args.output_dir, 
                                 save_intermediate=False,
                                 training_args=get_training_args(finetune_args))
    trainer.fit(model, dst_loaders['train'], dst_loaders['val'])

    # eval
    model.set_grad_mode(do_overall_model=False, do_classifier=False, do_sec_classifier=False)
    all_results = {}
    for stage, loader_dict, do_secondary in [
        ['pretrain', src_loaders, False],
        ['finetune', dst_loaders, True],
    ]:
        all_results[stage] = {}
        print(f"===== {stage} eval =====")
        model.do_secondary = do_secondary
        for split in ['val', 'test']:
            print(split)
            loader = loader_dict[split]
            all_results[stage][split] = evaluate_model(model, loader)

    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.save_checkpoint == 1:
        checkpoint_path = os.path.join(training_args.output_dir, f'checkpoint_last')
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
    if training_args.save_output == 1:
        torch.save(all_results, os.path.join(training_args.output_dir, "results.pt"))



if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (TrainingArguments, CounterfactualArguments)
    )
    training_args, cf_args = parser.parse_args_into_dataclasses()
    main(training_args, cf_args)
