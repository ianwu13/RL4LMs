"""
Train a decision transformer based on T5

"""
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from random import randint
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torch.optim import AdamW

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

output_dir = "/home/ICT2000/chawla/nego_rl/logs/offline_rl/dummy_1"

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.save_hyperparameters(hparams)
    
    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
  
  def is_logger(self):
    return self.trainer.global_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=lm_labels,
    )

  def _step(self, batch):
    lm_labels = batch["target_ids"]
    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        lm_labels=lm_labels,
        decoder_attention_mask=batch['target_mask']
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}

    self.log("avg_train_loss", avg_train_loss)
    self.log("log", tensorboard_logs)
    self.log("progress_bar", tensorboard_logs)

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.log("val_loss", loss)
  
  def validation_epoch_end(self, outputs):
    if outputs:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.log("avg_val_loss", avg_loss)
        self.log("log", tensorboard_logs)
        self.log("progress_bar", tensorboard_logs)

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]

  def optimizer_step(self,
                   epoch=None,
                   batch_idx=None,
                   optimizer=None,
                   optimizer_idx=None,
                   optimizer_closure=None,
                   on_tpu=None,
                  #  using_native_amp=None,
                   using_lbfgs=None
                   ):
    optimizer.step(closure=optimizer_closure) # The parameters in this can be removed   But not recommended 
    optimizer.zero_grad()
    self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=12)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="validation", args=self.hparams)
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=12)

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))

args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=32,
    eval_batch_size=32,
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

"""

1. create trajectories.
2. then create inputs/outputs

Sample a trajectory length l: 3 - 20 -> done
randomly choose l characters (with replacement) from english alphabet: a...z
compute final rewards:
    - +10 for every occurance of a (max 5 times is rewarded)
    - -2 for every character generated.
    - Best possible sequence would be a a a a a done = 50 - 10 = 40
"""

def get_io(this_traj, this_rtgs, target):
    inp = f"reward-to-go: {' '.join([str(arr) for arr in this_rtgs])} history: {' '.join([str(arr) for arr in this_traj])}"
    outp = target
    return inp, outp

def count_prev_as(lst):
    cnt = 0
    for item in lst:
        if item == 'a':
            cnt += 1
    return cnt

def get_rtgs(traj):

    rewards = []
    for i in range(len(traj)):
        score = 0
        if traj[i] != 'done':
            score += -2
        if traj[i] == 'a':
            prev_as = count_prev_as(traj[:i])
            if prev_as < 5:
                score += 10

        rewards.append(score)

    rtgs = []
    for i in range(len(rewards)):
        rtgs.append(sum(rewards[i:]))

    return rtgs 

raw_data = []
min_l, max_l = 3, 20
chars = ['a', 'b', 'c']# 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
size = 2000 #5000

for _ in range(size):
    l = random.randint(min_l, max_l)

    traj = []
    for _ in range(l):
        traj.append(random.choice(chars))

    traj.append('done')
    
    raw_data.append(traj)

io_data = []

for traj in raw_data:
    
    #rtg[i] is the rtg starting from the ith action. 
    rtgs = get_rtgs(traj)
    assert len(rtgs) == len(traj)

    ll = len(traj)

    for i in range(ll):
        # we will try to predict the i^th action.
        # rtg seq length will be of size (i+1)
        # traj seq length will be of size (i).

        this_traj = traj[:i][:]
        this_rtgs = rtgs[:i+1][:]
        target = traj[i]

        assert len(this_traj) == i
        assert len(this_rtgs) == i+1

        inp, outp = get_io(this_traj, this_rtgs, target)

        io_data.append((inp, outp))

print(io_data[:20])

random.shuffle(io_data)

raw_dataset = {
    "train": io_data[:int(0.8*len(io_data))],
    "validation": io_data[int(0.8*len(io_data)):],
}

raw_dataset["test"] = raw_dataset["validation"][:]

print(len(raw_dataset["train"]))

tokenizer = T5Tokenizer.from_pretrained('t5-base')

class CustomDataset(Dataset):
  def __init__(self, tokenizer, raw_dataset, type_path,  max_len=512):
    self.data = raw_dataset[type_path]
    
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    for idx in range(min(len(self.data), 1000000)):
      input_, target = self.data[idx][0], str(self.data[idx][1])
      
      input_ = input_
      target = target

       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [input_], max_length=self.max_len, padding="max_length", return_tensors="pt", truncation=True
      )
       # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=5, padding="max_length", return_tensors="pt", truncation=True
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

args_dict.update({'output_dir': output_dir})
args = argparse.Namespace(**args_dict)
print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    accelerator="gpu",
    devices=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    # amp_backend="apex",
    # amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    callbacks=[LoggingCallback(),checkpoint_callback],
)

def get_dataset(tokenizer, type_path, args):
  return CustomDataset(tokenizer, raw_dataset, type_path=type_path,  max_len=args.max_seq_length)

model = T5FineTuner(args)

trainer = pl.Trainer(**train_params)

trainer.fit(model)

#save model and tokenizer again
model.model.save_pretrained(output_dir)
model.tokenizer.save_pretrained(output_dir)