# GraphMERT
# This trains a Heterogeneous Graph Attention Transformer (GraphMERT) using PyTorch.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union

verbose = True

# -------------------------------------------------------------------------------------
# Load Corpus (sentences)
# -------------------------------------------------------------------------------------
from pathlib import Path
def get_corpus(filepath: Path) -> List[str]:
    # Split into sentences (a simple approach)
    # import nltk
    # nltk.download('punkt')
    import nltk
    nltk.download('punkt_tab')
    nltk.download('punkt')
    from nltk import sent_tokenize

    # Load large text corpus from a .txt file (e.g., downloaded from Project Gutenberg)
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    sentences = sent_tokenize(text)

    # Now you can use sentences as your corpus list instead of a small example list
    corpus = sentences
    return corpus



def get_embeddings(corpus):
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA


    # Load Sentence Transformer model
    # Use SBERT to get fixed-size embeddings
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    original_dim = sbert_model.get_sentence_embedding_dimension()  # Typically 384
    target_dim = 10  # as per your JSON structure

    # Step 1: Encode all sentences using the Sentence Transformer
    sbert_embeddings = sbert_model.encode(corpus)

    # Step 2: Reduce 384-dim embeddings to 10-dim using PCA
    n_components = min(target_dim, sbert_embeddings.shape[0], sbert_embeddings.shape[1])
    pca = PCA(n_components=target_dim)
    reduced_sbert_emb = pca.fit_transform(sbert_embeddings)
    return reduced_sbert_emb, sbert_model, pca

def build_graphs(reduced_embeddings):
    seq_len = 5
    target_dim = 10  # as per your JSON structure


    graphs = []
    num_graphs = (len(reduced_embeddings) + seq_len - 1) // seq_len

    for i in range(num_graphs):
        start = i * seq_len
        end = min(len(reduced_embeddings), (i + 1) * seq_len)
        chunk = reduced_embeddings[start:end].tolist()

        # Pad with zero vectors if less than seq_len
        while len(chunk) < seq_len:
            chunk.append([0.0] * target_dim)

        # Labels matching your example pattern (length=5)
        if i == 0:
            labels = [1, 0, 2, 1, 0]
        else:
            labels = [0, 1, 0, 2, 1]

        graphs.append({
            "features": chunk,
            "labels": labels
        })

    # Save final dataset JSON exactly like your example structure
    with open("content/sentence_transformer_fixed10d_graphs.json", "w") as f:
        json.dump(graphs, f, indent=2)

    if verbose: print(f"Saved {len(graphs)} graph examples to sentence_transformer_fixed10d_graphs.json")


# -------------------------------------------------------------------------------------
# run training
# -------------------------------------------------------------------------------------

# H-GAT Layer with consistent dims
class HeteroGraphAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim // num_heads, num_heads=1, batch_first=True)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x_proj = self.linear(x)
        splits = torch.chunk(x_proj, self.num_heads, dim=-1)
        head_outputs = []
        for split, attn in zip(splits, self.attention_heads):
            out, _ = attn(split, split, split)
            head_outputs.append(out)
        return torch.cat(head_outputs, dim=-1)

# GraphMERT Encoder
class GraphMERTEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads):
        super().__init__()
        self.total_embed_dim = embed_dim * num_heads
        self.embedding = nn.Linear(input_dim, self.total_embed_dim)
        self.hgat_layers = nn.ModuleList([
            HeteroGraphAttentionLayer(self.total_embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        for hgat in self.hgat_layers:
            x = hgat(x)
        x = self.transformer_encoder(x)
        return x
    
class ChainGraphDataset(Dataset):
    def __init__(self, json_path):
        import json
        with open(json_path, 'r') as f:
            self.chain_graphs = json.load(f)

    def __len__(self):
        return len(self.chain_graphs)

    def __getitem__(self, idx):
        example = self.chain_graphs[idx]
        features = torch.tensor(example["features"], dtype=torch.float)
        labels = example.get("labels")
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = torch.full((features.shape[0],), -100, dtype=torch.long)
        return features, labels


# Training loop with loss and optimizer
def train_graphmert(model, dataloader, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(num_epochs):
        total_loss = 0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)  # (batch, seq_len, embed_dim*num_heads)
            logits = outputs.view(-1, outputs.size(-1))
            loss = criterion(logits, labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose: print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")


def run_training():
    input_dim = 10
    embed_dim = 32
    num_heads = 4  # should be a divisor of total embedding dimension and even for multihead attention
    num_layers = 3
    batch_size = 8
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for Synthetic dataset matching model input/output shapes
      # dataset = ChainGraphDataset(num_samples=100, seq_len=5, feature_dim=input_dim, num_classes=10)
      # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset = ChainGraphDataset("content/sentence_transformer_fixed10d_graphs.json")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = GraphMERTEncoder(input_dim, embed_dim, num_layers, num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_graphmert(model, dataloader, optimizer, num_epochs, device)

    # Save the trained model weights (saves and loads only the model parameters)
    save_path = "content/graphmert_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    if verbose: print(f"Model saved to {save_path}")

    # Save the entire model
    save_path = "content/graphmert_entire_model.pth"
    torch.save(model, save_path)

    # optional: if fine-tuned head weights exist
    # torch.save(class_head.state_dict(), "graphmert_classifier_head.pth")
    # if verbose: print(f"class_head Model saved to graphmert_classifier_head.pth")


def main():
    # corpus is from Romeo and Juliet for free from gutenberg:
    # https://www.gutenberg.org/ebooks/1513.txt.utf-8

    # get corpus and graphs
    corpus = get_corpus(Path("content/pg1513.txt"))
    reduced_embeddings, sbert_model, pca = get_embeddings(corpus)
    build_graphs(reduced_embeddings)

    # run training
    run_training()


if __name__ == "__main__":
    # Windows-sichere Multiprocessing-Initialisierung
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()