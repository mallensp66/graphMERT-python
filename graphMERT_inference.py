from graphMERT_train_encoder import GraphMERTEncoder, get_corpus, get_embeddings, HeteroGraphAttentionLayer
import torch
from pathlib import Path
import numpy as np

verbose = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    only_load_weights = False
    # STEP 1: Load trained model weights
    
    input_dim = 10

    if(only_load_weights):
        embed_dim = 32
        num_heads = 4
        num_layers = 2

        model_path = "content/graphmert_model_weights.pth"
        model = GraphMERTEncoder(input_dim, embed_dim, num_layers, num_heads)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        model_path = "content/graphmert_entire_model.pth"
        model = torch.load(model_path, weights_only=False, map_location=device)
        model.eval()
    return model


def get_reduced_graphmert_emb(corpus, model, input_dim):
    # Use SBERT to get fixed-size embeddings
    reduced_sbert_emb, sbert_model, pca_text = get_embeddings(corpus)

    # --- Prepare inputs and run GraphMERT inference ---
    # STEP 4: Convert to tensor and shape (1 batch, 5 nodes, 10 features)
    input_features = torch.tensor([reduced_sbert_emb], dtype=torch.float32, device=device)  # Shape: (1,5,10)

    # STEP 5: Inference
    with torch.no_grad():
        graphmert_outputs = model(input_features)  # Shape: (1, 5, embed_dim * num_heads)
        if verbose: print("Inference output shape:", graphmert_outputs.shape)
        if verbose: print("Inference output tensor:", graphmert_outputs)

    # Assuming 'corpus' is your list of input sentences or text chunks fed to the model
    graphmert_embeddings = graphmert_outputs[0].cpu().numpy()  # shape: (3166, 128)

    for idx, embedding in enumerate(graphmert_embeddings):
        text_unit = corpus[idx]  # original text corresponding to node idx
        if verbose: print(f"Text unit #{idx}: {text_unit}")
        if verbose: print(f"Embedding vector sample: {embedding[:5]}...")  # show first 5 dims for brevity


    # --- Reduce GraphMERT embeddings with separate PCA ---
    from sklearn.decomposition import PCA
    pca_graphmert = PCA(n_components=input_dim)
    reduced_graphmert_emb = pca_graphmert.fit_transform(graphmert_embeddings)  # (N, 10)
    return reduced_graphmert_emb, sbert_model, pca_text

def build_graph(corpus, reduced_graphmert_emb):
    # --- Build NetworkX graph with text + GraphMERT PCA embeddings ---
    import networkx as nx
    G = nx.Graph()
    for idx, (txt, emb) in enumerate(zip(corpus, reduced_graphmert_emb)):
        G.add_node(idx, text=txt, embedding=emb)
    for i in range(len(corpus) - 1):
        G.add_edge(i, i + 1)
    return G

# --- Retrieval using SBERT + PCA embeddings (consistent with corpus) ---
def retrieve_similar_nodes(query, graph, sbert_model, pca_text, top_k=3):
    query_emb = sbert_model.encode([query])
    query_emb_reduced = pca_text.transform(query_emb)[0]  # Shape (10,)
    scores = []
    for node in graph.nodes:
        node_emb = graph.nodes[node]['embedding']  # GraphMERT PCA embeddings also 10-dim
        # Cosine similarity
        sim = np.dot(query_emb_reduced, node_emb) / (np.linalg.norm(query_emb_reduced) * np.linalg.norm(node_emb))
        scores.append((node, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [graph.nodes[n]['text'] for n, _ in scores[:top_k]]

# --- Setup generation pipeline (GPT-2 example) ---
# generator = pipeline('text-generation', model='Qwen/Qwen3-0.6B')
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')

def graphrag_generate(query, G, sbert_model, pca_text):

    retrieved = retrieve_similar_nodes(query, G, sbert_model, pca_text)
    context = "\n".join(retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    results = generator(prompt, max_length=150, num_return_sequences=1)
    return results[0]['generated_text']

def main():
    model = load_model()
    # STEP 2: Prepare input text corpus to embeddings
    # Load large text corpus from a .txt file (e.g., downloaded from Project Gutenberg)
    corpus = get_corpus(Path("content/pg1513.txt"))

    reduced_graphmert_emb, sbert_model, pca_text = get_reduced_graphmert_emb(corpus, model, 10)



    G = build_graph(corpus, reduced_graphmert_emb)
    # --- Example usage ---
    query = "What part of the story does romeo talk to the Friar about being banished?"
    answer = graphrag_generate(query, G, sbert_model, pca_text)
    print("GraphRAG answer:\n", answer)
    pass

if __name__ == "__main__":
    # Windows-sichere Multiprocessing-Initialisierung
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()