"""Interactively inspect up to three ETPC development examples at a time.

Run from the repository root:
    python paraphrase_generation/output_explorer.py --help

Requires the locally cached facebook/bart-large tokenizer/config and a state_dict
checkpoint saved by bart_generation.py. No training or BLEU evaluation is run.
"""

import argparse
import csv
from pathlib import Path

DEFAULT_DEV_PATH = Path(__file__).resolve().parents[1] / "data/etpc-paraphrase-dev.csv"
BASE_MODEL = "facebook/bart-large"
REQUIRED_COLUMNS = (
    "id", "sentence1", "sentence1_segment_location", "paraphrase_type_ids", "sentence2",
)
MAX_EXAMPLES = 3


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number



def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=Path, required=True,
        help="Path to a BART generation state_dict checkpoint (.pt).",
    )
    parser.add_argument("--dev_path", type=Path, default=DEFAULT_DEV_PATH)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max_length", type=positive_int, default=50,
        help="Maximum output token length, matching dev evaluation by default.",
    )
    parser.add_argument("--num_beams", type=positive_int, default=5)

    return parser.parse_args(argv)


def load_examples(path, ids=None, offset=0, limit=10):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Development CSV is missing columns: {', '.join(sorted(missing))}")
        rows = []
        for index, row in enumerate(reader):
            if any(row[column] is None or not row[column].strip() for column in REQUIRED_COLUMNS):
                raise ValueError(f"Missing required value in development row {index}.")
            rows.append({"dev_row": index, **{key: row[key] for key in REQUIRED_COLUMNS}})

    if ids is not None:
        requested = {value.strip().lower() for value in ids}
        available = {row["id"].strip().lower() for row in rows}
        if requested - available:
            raise ValueError(f"Unknown ETPC IDs: {', '.join(sorted(requested - available))}")
        rows = [row for row in rows if row["id"].strip().lower() in requested]
    else:
        rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No development examples selected. Check --offset and --dev_path.")
    return rows


def format_input(row):
    # Keep separators and field order identical to bart_generation.transform_data.
    return " </s> ".join(
        row[key] for key in ("sentence1", "sentence1_segment_location", "paraphrase_type_ids")
    )


def generate_examples(model, tokenizer, rows, device, batch_size=3, max_length=50, num_beams=5):
    if not 1 <= len(rows) <= MAX_EXAMPLES:
        raise ValueError("Select between 1 and 3 examples per request.")

    import torch

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs = [format_input(row) for row in batch]
            encoded = tokenizer(
                inputs, max_length=256, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            outputs = model.generate(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                max_length=max_length, num_beams=num_beams,
                do_sample=False, early_stopping=num_beams > 1,
            )
            predictions = tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )
            for row, model_input, prediction in zip(batch, inputs, predictions):
                yield {**row, "model_input": model_input, "generated_sentence2": prediction}


def print_example(row):
    print(f"\n{'=' * 80}\nDev row {row['dev_row']} | ETPC ID: {row['id']}")
    print(f"Source:     {row['sentence1']}")
    print(f"Segments:   {row['sentence1_segment_location']}")
    print(f"Type IDs:   {row['paraphrase_type_ids']}")
    print(f"Model input (before tokenization): {row['model_input']}")
    print(f"Generated:  {row['generated_sentence2']}")
    print(f"Reference:  {row['sentence2']}")


def select_rows(command, rows):
    tokens = command.replace(",", " ").split()
    if not 1 <= len(tokens) <= MAX_EXAMPLES:
        raise ValueError("Enter between 1 and 3 row numbers.")
    try:
        indices = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("Use row numbers, for example: 0 12 24.") from error
    if len(set(indices)) != len(indices):
        raise ValueError("Enter distinct row numbers.")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError(f"Row numbers must be between 0 and {len(rows) - 1}.")
    return [rows[index] for index in indices]


def explore_console(model, tokenizer, rows, device, max_length=50, num_beams=5):
    print(f"\nLoaded {len(rows)} dev examples (row numbers 0–{len(rows) - 1}).")
    print("Enter 1–3 row numbers separated by spaces or commas, e.g. 0 12 24.")
    print("Press Enter for the next three examples; enter q to quit.")
    next_row = 0
    while True:
        try:
            command = input("\nDev rows [Enter=next 3, q=quit]: ").strip()
            if command.lower() in {"q", "quit", "exit"}:
                break
            if command:
                try:
                    selected = select_rows(command, rows)
                except ValueError as error:
                    print(error)
                    continue
            else:
                if next_row >= len(rows):
                    print("End of the dev set. Choose row numbers to revisit examples, or q to quit.")
                    continue
                selected = rows[next_row:next_row + MAX_EXAMPLES]
                next_row += len(selected)
            print(f"Generating {len(selected)} example(s)...", flush=True)
            for record in generate_examples(
                model, tokenizer, selected, device,
                max_length=max_length, num_beams=num_beams,
            ):
                print_example(record)
        except (EOFError, KeyboardInterrupt):
            break
    print("\nGoodbye.")


def main(argv=None):
    args = get_args(argv)
    if not args.model_path.is_file():
        raise SystemExit(f"Checkpoint file not found: {args.model_path}")
    try:
        rows = load_examples(args.dev_path, limit=0)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    import torch
    from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu or run on a GPU node.")

    print(f"Checkpoint: {args.model_path}\nDevice: {device}\nExamples: {len(rows)}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
        config = BartConfig.from_pretrained(BASE_MODEL, local_files_only=True)
    except OSError as error:
        raise SystemExit(
            "The facebook/bart-large tokenizer/config is not cached locally. "
            "Run in the training environment after the repository's BART setup."
        ) from error
    if not isinstance(config, BartConfig):
        raise SystemExit("Expected a BART configuration for the generation checkpoint.")
    model = BartForConditionalGeneration(config)
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.to(device=torch.device(device))

    explore_console(model, tokenizer, rows, device, args.max_length, args.num_beams)


if __name__ == "__main__":
    main()
