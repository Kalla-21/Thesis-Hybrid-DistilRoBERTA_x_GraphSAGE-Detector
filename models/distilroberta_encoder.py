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

    @torch.no_grad()
    def embed_texts(self, texts, max_len=512, batch_size=16):
        """Returns a (len(texts), 768) float32 array. Padding to max_len keeps results identical across calls."""
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                [str(t) for t in texts[start:start + batch_size]],
                add_special_tokens=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.encoder(input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"]).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outputs.append(pooled.cpu().numpy())
        return np.vstack(outputs)