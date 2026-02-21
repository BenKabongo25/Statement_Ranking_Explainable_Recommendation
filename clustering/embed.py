import argparse
import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def main(args):
    sentence_df = pd.read_csv(args.input_path, index_col=0)
    sentences = sentence_df[args.sentence_column].apply(str).tolist()
    
    print(f"Number of sentences to embed: {len(sentences)}")
    print("Sample sentence:", sentences[0])

    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(
        sentences, 
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=False,
        convert_to_tensor=True,
        batch_size=args.batch_size
    )

    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Type of embeddings: {type(embeddings)}")
    print(f"Norm of first embedding vector: {torch.norm(embeddings[0], p=2).item()}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(embeddings, args.output_path)
    print(f"Embeddings saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to sentences."
    )
    parser.add_argument(
        "--sentence_column",
        type=str,
        default="sentence",
        help="Column name for sentences in the input."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save embeddings."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="Pre-trained model name."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for embedding generation."
    )
    
    args = parser.parse_args()
    main(args)