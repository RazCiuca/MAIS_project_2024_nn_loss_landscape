"""
We do line search for a bert model on wikipedia
"""

from datasets import load_dataset
from transformers import BertTokenizer, BertTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load dataset
logging.info("Loading the Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en")['train'].shuffle()
subset_dataset = dataset.select(range(20000))  # Using a subset for quick processing

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Function to tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256, return_tensors='pt')

# Tokenize dataset
logging.info("Tokenizing data...")
tokenized_datasets = subset_dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# DataLoader
logging.info("Setting up DataLoader...")
# casting to list makes the dataloader set the labels, so we're not evaluating different labels every time we call the network
train_dataloader = list(DataLoader(tokenized_datasets, batch_size=16, collate_fn=data_collator))

# Training model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

from torch.autograd.functional import vhp
from torch.func import functional_call
import torch as t
import numpy as np
import sys

params = [{name: p} for (name, p) in model.named_parameters()]

eigvals, eigvecs = t.load('./bert_stuff/smallest_eigs_10000_docs')

eigvals = eigvals.flatten()
eigvecs = eigvecs.flatten()

# ===========================================================================
# Doing the Line Search with the lowest Eigenvector
# ===========================================================================

# predicted displacement needed to decrease loss by 0.5
max_alpha = (2.0/np.abs(eigvals) * 0.5)**0.5
n_steps = 20
alphas_pos = np.arange(0, max_alpha, max_alpha/n_steps)
alphas_neg = np.arange(-max_alpha, 0, max_alpha/n_steps)[::-1]
alphas = np.concatenate([alphas_pos, alphas_neg])
losses = []

# shape the eigenvector as BERT params
line_search_dir = []
index = 0
for p in model.parameters():
    x_shape = p.shape
    n = np.prod(x_shape)
    line_search_dir.append(t.from_numpy(eigvecs[index:n + index].reshape(x_shape)).float().to(device))
    index += n

# do the line search
for alpha in alphas:

    with t.no_grad():

        loss_fn = torch.nn.CrossEntropyLoss()

        # define new model parameters at line search point
        new_params = []

        for v,p in zip(line_search_dir, list(model.parameters())):
            new_params.append(p + alpha * v)

        # functional call with those model params on the whole batch
        n_total_batch = len(train_dataloader)
        total_loss = 0
        n_batch = 0

        for batch in train_dataloader:

            n_batch += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            param_names = dict(model.named_parameters()).keys()

            z = [{name: p} for (p, name) in zip(new_params, param_names)]

            outputs = functional_call(model, z, (batch['input_ids'], batch['attention_mask']))
            loss = loss_fn(outputs.logits.flatten(0, 1), batch['labels'].flatten(0, 1))

            print(f" alpha:{alpha}, batch {n_batch}/{n_total_batch}, mean loss so far: {total_loss/n_batch}")
            print(f"losses: {losses}")

            total_loss += loss.item()

        total_loss /= n_total_batch
        losses.append(total_loss)

# saving and plotting the line search losses

losses = np.array(losses)

t.save((alphas, losses, eigvals, eigvecs), './bert_stuff/smallest_eigen_line_search')

import matplotlib.pyplot as plt

plt.plot(alphas, losses)
plt.xlabel('$alpha$')
plt.ylabel('loss')
plt.title(f'line search of pre-trained BERT on wikipedia mini-batch along eigval with $\lambda={eigvals[0]:.2f}$')
plt.show()



