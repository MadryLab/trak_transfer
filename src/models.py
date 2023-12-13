import torchvision
import torch
import torchvision.transforms as T
import numpy as np
from transformers import AutoImageProcessor
from transformers import PretrainedConfig, ResNetModel, ResNetPreTrainedModel
import torch.nn as nn
from typing import Optional
import copy


class CustomResNet(ResNetPreTrainedModel):
    def __init__(self, config, arch, num_src_labels, num_dst_labels):
        if config is None:
            config = PretrainedConfig.from_pretrained(arch)
        super().__init__(config)
        self.resnet = ResNetModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], num_src_labels)
        )
        if num_dst_labels == -1:
            self.secondary_classifier = None
        else:
            self.secondary_classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(config.hidden_sizes[-1], num_dst_labels),
            )
        # initialize weights and apply final processing
        self.post_init()
        self.do_secondary = False
        
    def set_grad_mode(self, do_overall_model, do_classifier, do_sec_classifier):
        print({
            'do_secondary': self.do_secondary, 
            'overall': do_overall_model, 
            'classifier': do_classifier, 
            'sec classifier': do_sec_classifier
        })
            
        if self.secondary_classifier is not None:
            for param in self.secondary_classifier.parameters():
                param.requires_grad = do_sec_classifier
        for param in self.resnet.parameters():
            param.requires_grad = do_overall_model
        for param in self.classifier.parameters():
            param.requires_grad = do_classifier
        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        if self.do_secondary:
            logits = self.secondary_classifier(pooled_output)
        else:
            logits = self.classifier(pooled_output)
        return logits