"""
Goal of this script:

- define Hessian-Vector product for models with non-twice-differentiable hessians
- Does BERT have negative eigenvalues when trained on the wikipedia dataset?
- find the top positive eigenvalues of BERT, estimate its spectrum.
- do line search in those directions, see how far away from minimum it is in those directions
- optimise only in that subspace and see what we get

- find top 5 eigenvalues and bottom 5 eigenvalues to within 1e-5 precision



"""

import torch as t
import numpy as np

# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from transformers import BertModel

from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling

if __name__ == "__main__":

    dataset = load_dataset("wikipedia", "20220301.en")

    # dataset.data['train'] is shape (6458670, 4), each containing an article

    model = BertModel.from_pretrained("google-bert/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    #
    # # This is new
    # batch["labels"] = torch.tensor([1, 1])
    #
    # optimizer = AdamW(model.parameters())
    # loss = model(**batch).loss


