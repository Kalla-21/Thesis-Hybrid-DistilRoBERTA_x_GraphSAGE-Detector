from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# Import your model architectures
from model.graphsage_model import FraudDetectionGraphSAGE
from model.distilroberta_encoder import DistilRobertaModule

app = FastAPI(title="Hybrid Scam Detection API")

# Allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL SETUP ---
TEXT_DIM = 768
TABULAR_DIM = 3 
DEVICE = "cpu" 

print("Loading models...")
scam_model = FraudDetectionGraphSAGE(text_dim=TEXT_DIM, tabular_dim=TABULAR_DIM, fusion_mode="cross_attention")
scam_model.load_state_dict(torch.load("model/best_cross_attention_model.pt", map_location=torch.device('cpu'), weights_only=True))
scam_model.eval()

# Load the DistilRoBERTa translator
text_encoder = DistilRobertaModule(DEVICE)
print("Models loaded successfully!")

# --- DATA SCHEMA (User Experience) ---
class JobInput(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str
    benefits: str
    telecommuting: int
    has_company_logo: int
    has_questions: int

# --- PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict_scam(job: JobInput):
    try:
        # 1. Convert Text to 768D Embedding (Matching utils.py string concatenation)
        combined_text = f"{job.title} {job.company_profile} {job.description} {job.requirements} {job.benefits}"
        text_embedding = text_encoder.extract_single_embedding(combined_text)
        x_text_tensor = torch.tensor(text_embedding, dtype=torch.float)

        # 2. Convert Tabular Data to 3D Array
        raw_tabular = [job.telecommuting, job.has_company_logo, job.has_questions] 
        x_tab_tensor = torch.tensor([raw_tabular], dtype=torch.float)

        # 3. Handle Graph Edges (Dummy self-loop for a single isolated prediction)
        edge_index_tensor = torch.tensor([[0], [0]], dtype=torch.long)

        # 4. Predict
        with torch.no_grad():
            logits = scam_model(x_text_tensor, edge_index_tensor, x_tab_tensor)
            probability = torch.sigmoid(logits).squeeze().item()
        
        return {"status": "success", "scam_probability": round(probability * 100, 2)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))