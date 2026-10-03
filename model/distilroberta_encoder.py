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
        # Classification head removed to save VRAM 
        self.encoder = AutoModel.from_pretrained(model_name).to(device)

    def extract_single_embedding(self, text: str, max_length: int = 256):
        """Converts a single text string into a 768-dim numpy array on the fly.

        max_length must match what the training embeddings used (JobTextDataset, 256).
        """
        self.encoder.eval()

        # 1. Tokenize the incoming string
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 2. Extract the embedding without tracking gradients
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Grab the [CLS] token embedding (the first token)
            embedding = outputs.last_hidden_state[:, 0, :] 
            
        return embedding.cpu().numpy()

    def extract_batch_embeddings(self, texts, max_length: int = 256, batch_size: int = 16):
        """Embeds many strings at once (used to re-score variants of a posting for explanations).

        Returns a (len(texts), 768) float tensor; same [CLS] embedding as extract_single_embedding.
        """
        self.encoder.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    texts[i:i + batch_size],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                ).to(self.device)
                hidden = self.encoder(**inputs).last_hidden_state
                out.append(hidden[:, 0, :].cpu())
        return torch.cat(out)

    def extract_embeddings(self, dataloader):
        self.encoder.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Nodes"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :] 
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)