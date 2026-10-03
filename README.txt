HOW TO GET STARTED:
1. Run this command to get started:

py -3.11 -m venv .venv ; .\.venv\Scripts\python.exe -m pip install -r requirements.txt ; .\.venv\Scripts\python.exe -m uvicorn main:app --reload

2. After doing that and it says "Application startup complete", right click index.html, then press "Open in integrated browser"

HOW TO USE:

1. Run this command to open server:

uvicorn main:app --reload

2. After doing that and it says "Application startup complete", right click index.html, then press "Open in integrated browser"

To push to own repo and bananalynn repo, commit using github desktop, then run this command:
git sync


HOW IT WORKS:

The page sends the posting to the local server, which scores it with the trained DistilRoBERTa + GraphSAGE
(cross-attention) model. The model was trained on one big graph of the dataset's job postings, so the server
loads that graph (model/graph_assets.npz) at startup and adds the new posting to it as an extra node, linked to
existing postings with the same company profile or the same title (same rules as training). Each request runs
the model over the whole graph, so expect about 2 seconds per prediction.

Files in model/ that the server needs:
- best_cross_attention_model.pt   trained weights
- graph_assets.npz                training graph (embeddings, edges, ...), made by build_graph_assets.py
- inference_config.json           decision threshold and the text length used in training

The result is a verdict plus a "model score". The score is NOT a probability: the model was trained with
class weighting and a tuned decision threshold, so a posting is flagged when score >= the threshold shown
(about 0.43), not when it passes 50%.

After a prediction the page shows a "Why this result?" panel (the /explain endpoint). It re-scores the posting
with one thing changed at a time and shows how much the score moves (positive = pushes toward scam):
- each field removed in turn (title, company profile, description, requirements, benefits)
- each sentence removed, as red/green highlighting (text beyond the model's 256-token reading window is greyed
  out: it is never read, so it can't affect the score)
- each yes/no setting flipped
- the graph: score with vs without the linked postings, plus a picture of the linked postings
These are counterfactual explanations of what moves the score, not a look inside the network. Expect a few
seconds, longer for postings that link to a big company's many postings.

The neighbour picture can show titles and scam/legit colours if model/graph_assets.npz has them. Add them with:
python add_neighbour_info.py <path to the ORIGINAL data/fake_job_postings.csv, 17,880 rows>
(it refuses a CSV that doesn't match the saved graph). Without it the picture still works, just uncoloured.

Linking only works on an exact match of the company profile / title text, so paste them exactly as they appear.
Postings that come from the original dataset are already in the training graph, so testing with those looks
better than the model really is on unseen postings.

To rebuild graph_assets.npz / inference_config.json (only needed if the weights or training data change),
run with the training repo's python:
<train repo>\.venv\Scripts\python.exe build_graph_assets.py --train-repo <train repo>


NOTE ON REPORTED ACCURACY:

The shipped weights are the model from CV fold 4 (of 5), chosen because it had the best cross-attention F1 on
its own test fold (0.827). Picking the best fold by its test score is slightly optimistic, so do not quote
0.827 as the model's accuracy. Quote the 5-fold mean instead: F1 0.812 (see outputs/cv_results.json in the
training repo).
