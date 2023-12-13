import os

from threading import Lock
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.optimizers import get_optimizer_and_lr_scheduler

lock = Lock()

class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val, sz):
        self.num += val*sz
        self.tot += sz

    def calculate(self):
        return self.num/self.tot

class LightWeightTrainer():
    def __init__(self, training_dir, save_intermediate, training_args):
        self.training_dir = training_dir        
        self.training_args = training_args
        self.check_training_args_()
        self.ce_loss = nn.CrossEntropyLoss()
        self.save_intermediate = save_intermediate


    def check_training_args_(self):
        for z in ['epochs', 'lr', 'weight_decay', 'momentum', 'lr_scheduler',
                'step_size', 'gamma', 'lr_milestones', 'lr_peak_epoch', 'eval_epochs']:
            assert z in self.training_args, f'{z} not in training_args'

    def get_accuracy(self, logits, target):
        correct = logits.argmax(-1) == target
        return (correct.float().mean()) * 100

    def get_opt_scaler_scheduler(self, model, iters_per_epoch=1):
        opt, scheduler = get_optimizer_and_lr_scheduler(self.training_args, model, iters_per_epoch)
        self.per_epoch_lr_scheduler = self.training_args['lr_scheduler'] !=  'cyclic'

        scaler = GradScaler()
        return opt, scaler, scheduler

    def training_step(self, model, batch):
        x, y = batch
        with lock:
            logits = model(x)
        loss = self.ce_loss(logits, y)
        acc = self.get_accuracy(logits, y)
        return loss, acc, len(x)

    def validation_step(self, model, batch):
        x, y = batch
        with lock:
            logits = model(x)
        loss = self.ce_loss(logits, y)
        acc = self.get_accuracy(logits, y)
        return loss, acc, len(x)

    def train_epoch(self, epoch_num, model, train_dataloader, opt, scaler, scheduler):
        #model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        with tqdm(train_dataloader) as t:
            t.set_description(f"Train Epoch: {epoch_num}")
            for batch in t:
                opt.zero_grad(set_to_none=True)
                with autocast():
                    loss, acc, sz = self.training_step(model, batch)
                t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                loss_meter.update(loss.item(), sz)
                acc_meter.update(acc.item(), sz)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if not self.per_epoch_lr_scheduler:
                    scheduler.step()
            if self.per_epoch_lr_scheduler:
                scheduler.step()
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc

    def val_epoch(self, epoch_num, model, val_dataloader):
        #model.eval()
        is_train = model.training
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        with torch.no_grad():
            with tqdm(val_dataloader) as t:
                t.set_description(f"Val Epoch: {epoch_num}")
                for batch in t:
                    with autocast():
                        loss, acc, sz = self.validation_step(model, batch)
                    t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                    loss_meter.update(loss.item(), sz)
                    acc_meter.update(acc.item(), sz)
                    
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        model.train(is_train)
        return avg_loss, avg_acc


    def fit(self, model, train_dataloader, val_dataloader):
        opt, scaler, scheduler = self.get_opt_scaler_scheduler(model, iters_per_epoch=len(train_dataloader))
        for epoch in range(self.training_args['epochs']):
            train_loss, train_acc = self.train_epoch(epoch, model, train_dataloader, opt, scaler, scheduler)
            curr_lr = scheduler.get_last_lr()[0]

            is_val_epoch = (epoch % self.training_args['eval_epochs'] == 0 and epoch != 0) or (epoch == self.training_args['epochs'] - 1)

            if is_val_epoch:
                val_loss, val_acc  = self.val_epoch(epoch, model, val_dataloader)
                print(f"LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}, Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}")
            else:
                print(f"LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}")

            # Save Checkpoints
            if is_val_epoch and self.save_intermediate:
                checkpoint_path = os.path.join(self.training_dir, f'checkpoint_{epoch}')
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)

