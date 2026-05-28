# Data Processing
python build_sequences.py --input-glob "Dataset/train/train_*.txt" --output-dir "processed_sequences"


# Exercise Embedding
python embed_exercises.py --input-path "processed_sequences/all_exercises.jsonl" --output-dir "processed_sequences/embeddings"


# Recommender Evaluation
python recommend_eval.py --model embedding_similarity --k-values 5,10


# Train and evaluate SASRec
python train_sasrec.py --epochs 5 --batch-size 64 --max-len 30 --use-pretrained-embeddings --use-outcome-embedding --k-values 5,10 --save-dir sasrec_semantic_outcome