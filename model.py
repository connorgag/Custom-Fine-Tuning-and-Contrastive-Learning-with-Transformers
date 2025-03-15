import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from peft import get_peft_model, LoraConfig, TaskType

class ScenarioModel(nn.Module):
    def __init__(self, args, tokenizer, target_size, lora=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args, lora)
        self.target_size = target_size
        self.projection = nn.Linear(768, args.embed_dim) 

        print(f"drop rate is {args.drop_rate}")
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args, self.target_size)

    def model_setup(self, args, lora):
        print(f"Setting up {args.model} model")
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        if lora:
          print("Trainable parameters before LoRA")
          print_trainable_params(self.encoder)

          # Create LoRA configuration
          lora_config = LoraConfig(
              r=args.rank,                # Rank of the LoRA decomposition
              lora_alpha=32,      # Alpha scaling factor for LoRA
              lora_dropout=0.1,   # dropout probability for LoRA
              bias="none",
              task_type="FEATURE_EXTRACTION",
          )
          
          self.encoder = get_peft_model(self.encoder, lora_config)
          
          print("Trainable parameters after LoRA")
          print_trainable_params(self.encoder)
              

    def forward(self, inputs, targets):
      outputs = self.encoder(**inputs)
      hidden = outputs.last_hidden_state[:,0,:]
      hidden = self.dropout(hidden)
      hidden = self.projection(hidden) 
      logit = self.classify(hidden)
      return logit
      

class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    print("Input dims: ",input_dim)
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size, n_layers_to_reinitialize):
    super().__init__(args, tokenizer, target_size)
    self._reinitialize_layers(n_layers_to_reinitialize)

  def _reinitialize_layers(self, n_layers_to_reinitialize):
    for n in range(1, n_layers_to_reinitialize + 1):
      self.encoder.encoder.layer[-n].apply(self._init_layer)

  def _init_layer(self, module):                        
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
      if module.bias is not None:
          module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)  


class SupConModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
    self.head = nn.Linear(feat_dim, feat_dim)
 
  def forward(self, inputs, targets):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    outputs = self.encoder(**inputs)
    hidden = outputs.last_hidden_state[:,0,:]
    hidden = self.dropout(hidden)
    hidden =  F.normalize(hidden,dim=1)
    head = self.head(hidden)
    return head


class EncoderWrapper(nn.Module):
    """Extracts only the encoder from SupConModel"""
    def __init__(self, supcon_model):
        super().__init__()
        self.encoder = supcon_model.encoder  # Direct access to original encoder
    
    def forward(self, inputs):
        # Returns only the <CLS> token representation
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]

class ClassifierWrapper(nn.Module):
    def __init__(self, supcon_model, classifier):
        super().__init__()
        # Create encoder-only wrapper
        self.base_model = EncoderWrapper(supcon_model)
        self.classifier = classifier
    
    def forward(self, inputs, labels=None):
        with torch.no_grad():
            features = self.base_model(inputs)  # Now returns raw <CLS> embeddings
        return self.classifier(features)


def print_trainable_params(peft_model):
  trainable_params = 0
  all_param = 0

  #iterating over all parameters
  for _, param in peft_model.named_parameters():
      #adding parameters to total
      all_param += param.numel()
      #adding parameters to trainable if they require a graident
      if param.requires_grad:
          trainable_params += param.numel()

  #printing results
  print(f"trainable params: {trainable_params:,}")
  print(f"all params: {all_param:,}")
  print(f"trainable: {100 * trainable_params / all_param:.2f}%")