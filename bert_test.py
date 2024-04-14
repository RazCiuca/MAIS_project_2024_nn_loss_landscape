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
subset_dataset = dataset.select(range(10000))  # Using a subset for quick processing

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
train_dataloader = DataLoader(tokenized_datasets, batch_size=16, collate_fn=data_collator)

# Training model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

from torch.autograd.functional import vhp
from torch.func import functional_call
import torch as t
import numpy as np
import sys

params = [{name: p} for (name, p) in model.named_parameters()]
total_hvp_counter = 0

def hvp_bert(tangents_np, model, dataloader):
    tangents_vec = t.from_numpy(tangents_np).to('cuda').squeeze()

    global total_hvp_counter
    total_hvp_counter += 1

    # shape tangents like params
    tangents = []
    index = 0

    for p in model.parameters():
        x_shape = p.shape
        n = np.prod(x_shape)
        tangents.append(tangents_vec[index:n + index].reshape(x_shape))
        index += n

    tangents = tuple(tangents)

    # res = np.zeros(tangents_np.shape[0])
    res = t.zeros(tangents_np.shape[0], device=device)

    param_names = dict(model.named_parameters()).keys()
    loss_fn = torch.nn.CrossEntropyLoss()

    n_batch = 0
    n_total_batch = len(dataloader)

    for batch in dataloader:
        n_batch += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        # sys.stdout.write(f"\r {total_hvp_counter}, doing batch {n_batch}/{n_total_batch}")
        print(f"\r {total_hvp_counter}, doing batch {n_batch}/{n_total_batch}")

        def fn_to_optim(*x):
            z = [{name: p} for (p, name) in zip(x, param_names)]

            outputs = functional_call(model, z, (batch['input_ids'], batch['attention_mask']))
            loss = loss_fn(outputs.logits.flatten(0,1), batch['labels'].flatten(0,1))
            return loss

        hessian_col = vhp(fn_to_optim, tuple(model.parameters()), tangents)[1]

        res += t.cat([x.flatten().detach() for x in hessian_col])

    return res.cpu().numpy()/n_total_batch

# testing hvp_bert
# n_params = sum([np.prod(x.shape) for x in model.parameters()])
# print(f"total number of parameters: {n_params}")
# tangents = np.random.randn(n_params)
# tangents /= tangents.std()
#
# result = hvp_bert(tangents, model, train_dataloader)

from scipy.sparse.linalg import LinearOperator, eigsh

def top_k_hessian_eigen_bert(model, dataloader, top_k = 100, mode='LA',v0=None, tol=1e-3):
    """
    computes top-k eigenvalues and eigenvectors of the hessian of model, with given data,
    possibly with finite batch size
    """
    n_params = sum([np.prod(x.shape) for x in model.parameters()])
    linop = LinearOperator((n_params, n_params),
                            matvec = lambda x: hvp_bert(x, model, dataloader))

    eigvals, eigvecs = eigsh(linop, k=top_k, which=mode, v0=v0, tol=tol)

    return eigvals, eigvecs


eigvals, eigvecs = top_k_hessian_eigen_bert(model, train_dataloader, top_k=1, mode='SA')

t.save((eigvals, eigvecs), './bert_stuff/smallest_eigs_10000_docs')

print(f"smallest eigenval for BERT: {eigvals}")
