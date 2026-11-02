# distilroberta_encoder.py
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


class DistilRobertaModule(nn.Module):
    """Frozen DistilRoBERTa used as an offline feature extractor (mean-pooled last hidden state)."""

    def __init__(self, device, model_name="distilroberta-base"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.encoder.eval()

    @torch.no_grad() # freezes the model parameters and disables gradient computation, which reduces memory usage and speeds up inference.
    def embed_texts(self, texts, max_len=512, batch_size=16): 
        # every 16 records is one batch, and the maximum length of the input sequence is 512 tokens. The function returns a numpy array of 
        # shape (len(texts), 768) containing the mean-pooled last hidden state embeddings for each input text.
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                [str(t) for t in texts[start:start + batch_size]],
                add_special_tokens=True, #tokenization process adds special tokens to the input text, 
                #such as [CLS] and [SEP], which are used by the model to indicate the beginning and end of a sequence.
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.encoder(input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"]).last_hidden_state #this gets one vector for each token in the input sequence, 
                                                                      # and the shape of hidden is (batch_size, sequence_length, hidden_size), where hidden_size is 768 for DistilRoBERTa.
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9) #mean pooling operation is performed on the hidden states of the tokens in the input sequence, weighted by the attention mask.
            #pooling results in a 768-dimensional vector for each input text, which is the mean of the hidden states of the tokens in the input sequence.
            outputs.append(pooled.cpu().numpy())
        return np.vstack(outputs) #put distilroberta embeddings into a single numpy array of shape (len(texts), 768) in outputs folder to be used and read later.