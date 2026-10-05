# distilroberta_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm.auto import tqdm

class DistilRobertaModule(nn.Module):
    def __init__(self, device, model_name="distilroberta-base"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)

    def extract_embeddings(self, dataloader):
        self.encoder.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Nodes"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                mask = attention_mask.unsqueeze(-1).float()
                embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)