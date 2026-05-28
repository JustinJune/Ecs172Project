from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
from io import open
import json
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build exercise-level user sequences from Duolingo SLAM raw files."
    )
    parser.add_argument(
        "--input-glob",
        default="Dataset/train/train_*.txt",
        help="Glob pattern for input SLAM files.",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_sequences",
        help="Directory where JSONL outputs will be written.",
    )
    parser.add_argument(
        "--outcome-mode",
        choices=["any_error", "error_rate"],
        default="any_error",
        help="How to derive one exercise-level outcome from token labels.",
    )
    return parser.parse_args()


def is_training_file(path):
    return "train" in os.path.basename(path).lower()


def parse_metadata_line(line, exercise_meta):
    if "prompt" in line:
        exercise_meta["prompt"] = line.split(":", 1)[1]
        return

    for pair in line[2:].split():
        key, value = pair.split(":", 1)
        if key == "countries":
            value = value.split("|")
        elif key == "days":
            value = float(value)
        elif key == "time":
            value = None if value == "null" else int(value)
        exercise_meta[key] = value


def finalize_exercise(exercise_meta, token_rows, training, source_file, source_index, outcome_mode):
    if not token_rows:
        return None

    first_instance_id = token_rows[0]["instance_id"]
    exercise_id = first_instance_id[:10]
    labels = [row["label"] for row in token_rows if "label" in row]
    token_text = [row["token"] for row in token_rows]

    record = {
        "exercise_uid": source_file + ":" + exercise_id,
        "exercise_id": exercise_id,
        "session_id": first_instance_id[:8],
        "user": exercise_meta["user"],
        "prompt": exercise_meta.get("prompt"),
        "tokens": token_text,
        "response_text": " ".join(token_text),
        "token_count": len(token_rows),
        "part_of_speech": [row["part_of_speech"] for row in token_rows],
        "dependency_labels": [row["dependency_label"] for row in token_rows],
        "countries": exercise_meta.get("countries", []),
        "days": exercise_meta.get("days"),
        "time": exercise_meta.get("time"),
        "client": exercise_meta.get("client"),
        "session": exercise_meta.get("session"),
        "format": exercise_meta.get("format"),
        "source_file": source_file,
        "source_exercise_index": source_index,
        "sort_key": [
            exercise_meta.get("days", math.inf),
            exercise_meta.get("time") if exercise_meta.get("time") is not None else math.inf,
            source_index,
        ],
    }

    if training:
        error_rate = sum(labels) / len(labels) if labels else 0.0
        record["token_labels"] = labels
        record["error_rate"] = error_rate
        record["any_error"] = int(any(label >= 0.5 for label in labels))
        record["all_correct"] = int(all(label < 0.5 for label in labels))
        record["exercise_outcome"] = (
            record["any_error"] if outcome_mode == "any_error" else error_rate
        )

    return record


def parse_slam_file(path, outcome_mode):
    exercises = []
    training = is_training_file(path)
    exercise_meta = {}
    token_rows = []
    source_index = 0

    with open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                record = finalize_exercise(
                    exercise_meta,
                    token_rows,
                    training,
                    os.path.basename(path),
                    source_index,
                    outcome_mode,
                )
                if record is not None:
                    exercises.append(record)
                    source_index += 1
                exercise_meta = {}
                token_rows = []
                continue

            if line[0] == "#":
                parse_metadata_line(line, exercise_meta)
                continue

            parts = line.split()
            token_row = {
                "instance_id": parts[0],
                "token": parts[1],
                "part_of_speech": parts[2],
                "dependency_label": parts[4],
            }
            if training:
                token_row["label"] = float(parts[6])
            token_rows.append(token_row)

    record = finalize_exercise(
        exercise_meta,
        token_rows,
        training,
        os.path.basename(path),
        source_index,
        outcome_mode,
    )
    if record is not None:
        exercises.append(record)

    return exercises


def build_user_sequences(exercises):
    by_user = defaultdict(list)
    for exercise in exercises:
        by_user[exercise["user"]].append(exercise)

    user_sequences = {}
    for user, items in by_user.items():
        user_sequences[user] = sorted(
            items,
            key=lambda item: (
                item["sort_key"][0],
                item["sort_key"][1],
                item["source_file"],
                item["sort_key"][2],
                item["exercise_id"],
            ),
        )
    return user_sequences


def build_leave_one_out_splits(user_sequences):
    train = []
    validation = []
    test = []
    sequence_lengths = Counter()

    for user, sequence in user_sequences.items():
        length = len(sequence)
        sequence_lengths[length] += 1

        if length == 1:
            test.append(sequence[-1])
        elif length == 2:
            validation.append(sequence[-2])
            test.append(sequence[-1])
        else:
            train.extend(sequence[:-2])
            validation.append(sequence[-2])
            test.append(sequence[-1])

    return train, validation, test, sequence_lengths


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_jsonl(path, rows):
    with open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise ValueError("No files matched input glob: " + args.input_glob)

    all_exercises = []
    for path in input_paths:
        all_exercises.extend(parse_slam_file(path, args.outcome_mode))

    user_sequences = build_user_sequences(all_exercises)
    train, validation, test, sequence_lengths = build_leave_one_out_splits(user_sequences)

    ensure_dir(args.output_dir)
    write_jsonl(os.path.join(args.output_dir, "all_exercises.jsonl"), all_exercises)
    write_jsonl(
        os.path.join(args.output_dir, "user_sequences.jsonl"),
        [{"user": user, "sequence": sequence} for user, sequence in sorted(user_sequences.items())],
    )
    write_jsonl(os.path.join(args.output_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.output_dir, "validation.jsonl"), validation)
    write_jsonl(os.path.join(args.output_dir, "test.jsonl"), test)

    print("Processed files:", len(input_paths))
    print("Exercises:", len(all_exercises))
    print("Users:", len(user_sequences))
    print("Train records:", len(train))
    print("Validation records:", len(validation))
    print("Test records:", len(test))
    print("Outcome mode:", args.outcome_mode)
    print("Sequence length histogram (first 10 lengths):")
    for length, count in sorted(sequence_lengths.items())[:10]:
        print("  length=%d users=%d" % (length, count))


if __name__ == "__main__":
    main()
