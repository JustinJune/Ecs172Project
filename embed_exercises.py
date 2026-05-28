from __future__ import annotations

import argparse
from io import open
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create semantic embeddings for Duolingo exercise texts."
    )
    parser.add_argument(
        "--input-path",
        default="processed_sequences/all_exercises.jsonl",
        help="Exercise-level JSONL built by build_sequences.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_sequences/embeddings",
        help="Directory for embedding artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face encoder name or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used during encoding.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max length.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of exercises to read for a quick smoke test.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load a cached model; do not try to download from Hugging Face.",
    )
    return parser.parse_args()


def load_exercises(path, limit=None):
    exercises = []
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                exercises.append(json.loads(line))
                if limit is not None and len(exercises) >= limit:
                    break
    return exercises


def choose_exercise_text(exercise):
    prompt = exercise.get("prompt")
    if prompt:
        return prompt, "prompt"
    return exercise["response_text"], "response_text"


def build_item_catalog(exercises):
    item_lookup = {}
    item_rows = []
    exercise_rows = []

    for exercise in exercises:
        text, text_source = choose_exercise_text(exercise)
        item_key = text.strip()
        if item_key not in item_lookup:
            item_id = "item_%07d" % len(item_rows)
            item_lookup[item_key] = item_id
            item_rows.append(
                {
                    "item_id": item_id,
                    "text": item_key,
                    "text_source": text_source,
                }
            )
        exercise_rows.append(
            {
                "exercise_uid": exercise["exercise_uid"],
                "exercise_id": exercise["exercise_id"],
                "user": exercise["user"],
                "item_id": item_lookup[item_key],
                "exercise_outcome": exercise.get("exercise_outcome"),
                "days": exercise.get("days"),
                "time": exercise.get("time"),
            }
        )

    return item_rows, exercise_rows


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(texts, model_name, batch_size, max_length, local_files_only):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

            end = min(start + batch_size, len(texts))
            print("Encoded %d/%d texts..." % (end, len(texts)))

    return np.vstack(all_embeddings)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_jsonl(path, rows):
    with open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    exercises = load_exercises(args.input_path, limit=args.limit)
    item_rows, exercise_rows = build_item_catalog(exercises)
    texts = [row["text"] for row in item_rows]

    print("Loaded exercises:", len(exercises))
    print("Unique items:", len(item_rows))
    print("Embedding model:", args.model_name)

    embeddings = encode_texts(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )

    ensure_dir(args.output_dir)
    write_jsonl(os.path.join(args.output_dir, "items.jsonl"), item_rows)
    write_jsonl(os.path.join(args.output_dir, "exercise_item_map.jsonl"), exercise_rows)
    np.savez_compressed(
        os.path.join(args.output_dir, "item_embeddings.npz"),
        item_ids=np.array([row["item_id"] for row in item_rows]),
        embeddings=embeddings,
    )

    print("Saved items to", os.path.join(args.output_dir, "items.jsonl"))
    print("Saved exercise-item map to", os.path.join(args.output_dir, "exercise_item_map.jsonl"))
    print("Saved embeddings to", os.path.join(args.output_dir, "item_embeddings.npz"))
    print("Embedding shape:", embeddings.shape)


if __name__ == "__main__":
    main()
