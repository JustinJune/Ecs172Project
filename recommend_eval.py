from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import open
import json
import math

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate next-exercise recommendation baselines with HR@K and NDCG@K."
    )
    parser.add_argument(
        "--train-path",
        default="processed_sequences/train.jsonl",
        help="Exercise-level training split.",
    )
    parser.add_argument(
        "--validation-path",
        default="processed_sequences/validation.jsonl",
        help="Exercise-level validation split.",
    )
    parser.add_argument(
        "--test-path",
        default="processed_sequences/test.jsonl",
        help="Exercise-level test split.",
    )
    parser.add_argument(
        "--k-values",
        default="5,10",
        help="Comma-separated cutoff values, e.g. 5,10",
    )
    parser.add_argument(
        "--model",
        choices=["popularity", "embedding_similarity"],
        default="popularity",
        help="Recommendation baseline to evaluate.",
    )
    parser.add_argument(
        "--items-path",
        default="processed_sequences/embeddings/items.jsonl",
        help="Unique item catalog from embed_exercises.py.",
    )
    parser.add_argument(
        "--embeddings-path",
        default="processed_sequences/embeddings/item_embeddings.npz",
        help="Compressed item embeddings from embed_exercises.py.",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_embedding_artifacts(items_path, embeddings_path):
    item_rows = load_jsonl(items_path)
    payload = np.load(embeddings_path)
    item_ids = payload["item_ids"]
    embeddings = payload["embeddings"].astype(np.float32)

    id_to_text = {}
    for row in item_rows:
        id_to_text[row["item_id"]] = row["text"]

    if len(item_ids) != len(embeddings):
        raise ValueError("Mismatch between item_ids and embeddings rows.")

    text_to_embedding = {}
    for item_id, embedding in zip(item_ids, embeddings):
        key = item_id.item() if hasattr(item_id, "item") else item_id
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        text = id_to_text[key]
        text_to_embedding[text] = embedding

    return {
        "item_rows": item_rows,
        "item_ids": item_ids,
        "embeddings": embeddings,
        "text_to_embedding": text_to_embedding,
        "all_item_texts": [row["text"] for row in item_rows],
    }


def choose_item_text(exercise):
    prompt = exercise.get("prompt")
    if prompt:
        return prompt.strip()
    return exercise["response_text"].strip()


def group_by_user(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["user"]].append(row)
    return grouped


def sort_sequence(rows):
    return sorted(
        rows,
        key=lambda row: (
            row["sort_key"][0],
            row["sort_key"][1],
            row["source_file"],
            row["source_exercise_index"],
            row["exercise_id"],
        ),
    )


def build_user_histories(train_rows, validation_rows):
    grouped = defaultdict(list)
    for row in train_rows:
        grouped[row["user"]].append(row)
    for row in validation_rows:
        grouped[row["user"]].append(row)
    return {user: sort_sequence(rows) for user, rows in grouped.items()}


def fit_popularity_model(train_rows, validation_rows):
    counts = Counter()
    for row in train_rows + validation_rows:
        counts[choose_item_text(row)] += 1
    ranked_items = [item for item, _ in counts.most_common()]
    return {
        "item_counts": counts,
        "ranked_items": ranked_items,
    }


def fit_embedding_similarity_model(train_rows, validation_rows, embedding_state):
    counts = Counter()
    for row in train_rows + validation_rows:
        counts[choose_item_text(row)] += 1

    candidate_texts = [
        item_text
        for item_text in embedding_state["all_item_texts"]
        if item_text in embedding_state["text_to_embedding"]
    ]

    return {
        "candidate_texts": candidate_texts,
        "text_to_embedding": embedding_state["text_to_embedding"],
        "popularity_counts": counts,
    }


def recommend_popularity(model_state, user_history, k):
    seen_items = {choose_item_text(row) for row in user_history}
    ranked = []
    for item in model_state["ranked_items"]:
        if item not in seen_items:
            ranked.append(item)
            if len(ranked) >= k:
                break
    return ranked


def mean_history_embedding(user_history, text_to_embedding):
    history_vectors = []
    for row in user_history:
        item_text = choose_item_text(row)
        if item_text in text_to_embedding:
            history_vectors.append(text_to_embedding[item_text])

    if not history_vectors:
        return None

    user_vector = np.mean(np.vstack(history_vectors), axis=0)
    norm = np.linalg.norm(user_vector)
    if norm == 0.0:
        return None
    return user_vector / norm


def recommend_embedding_similarity(model_state, user_history, k):
    seen_items = {choose_item_text(row) for row in user_history}
    user_vector = mean_history_embedding(user_history, model_state["text_to_embedding"])

    if user_vector is None:
        return []

    scored_items = []
    for item_text in model_state["candidate_texts"]:
        if item_text in seen_items:
            continue
        item_vector = model_state["text_to_embedding"][item_text]
        score = float(np.dot(user_vector, item_vector))
        popularity = model_state["popularity_counts"].get(item_text, 0)
        scored_items.append((score, popularity, item_text))

    scored_items.sort(reverse=True)
    return [item_text for _, _, item_text in scored_items[:k]]


def compute_hit_rate(target_item, ranked_items, k):
    return 1.0 if target_item in ranked_items[:k] else 0.0


def compute_ndcg(target_item, ranked_items, k):
    top_k = ranked_items[:k]
    try:
        rank = top_k.index(target_item)
    except ValueError:
        return 0.0
    return 1.0 / math.log(rank + 2, 2)


def evaluate_model(model_name, model_state, user_histories, test_rows, k_values):
    test_by_user = group_by_user(test_rows)
    metrics = {k: {"HR": 0.0, "NDCG": 0.0} for k in k_values}
    evaluated_users = 0

    for user, user_test_rows in test_by_user.items():
        history = user_histories.get(user, [])
        if not history:
            continue

        # Leave-one-out gives us one held-out test record per user.
        target = choose_item_text(sort_sequence(user_test_rows)[-1])
        max_k = max(k_values)

        if model_name == "popularity":
            ranked_items = recommend_popularity(model_state, history, max_k)
        elif model_name == "embedding_similarity":
            ranked_items = recommend_embedding_similarity(model_state, history, max_k)
        else:
            raise ValueError("Unsupported model: " + model_name)

        for k in k_values:
            metrics[k]["HR"] += compute_hit_rate(target, ranked_items, k)
            metrics[k]["NDCG"] += compute_ndcg(target, ranked_items, k)
        evaluated_users += 1

    if evaluated_users == 0:
        raise ValueError("No users could be evaluated.")

    for k in k_values:
        metrics[k]["HR"] /= evaluated_users
        metrics[k]["NDCG"] /= evaluated_users

    return metrics, evaluated_users


def main():
    args = parse_args()
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]

    train_rows = load_jsonl(args.train_path)
    validation_rows = load_jsonl(args.validation_path)
    test_rows = load_jsonl(args.test_path)

    user_histories = build_user_histories(train_rows, validation_rows)

    if args.model == "popularity":
        model_state = fit_popularity_model(train_rows, validation_rows)
    elif args.model == "embedding_similarity":
        embedding_state = load_embedding_artifacts(
            items_path=args.items_path,
            embeddings_path=args.embeddings_path,
        )
        model_state = fit_embedding_similarity_model(
            train_rows,
            validation_rows,
            embedding_state,
        )
    else:
        raise ValueError("Unsupported model: " + args.model)

    metrics, evaluated_users = evaluate_model(
        model_name=args.model,
        model_state=model_state,
        user_histories=user_histories,
        test_rows=test_rows,
        k_values=k_values,
    )

    print("Model:", args.model)
    print("Evaluated users:", evaluated_users)
    for k in k_values:
        print(
            "HR@%d=%.4f\tNDCG@%d=%.4f"
            % (k, metrics[k]["HR"], k, metrics[k]["NDCG"])
        )


if __name__ == "__main__":
    main()
