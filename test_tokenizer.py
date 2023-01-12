from transformers import PreTrainedTokenizerFast
import numpy as np
import argparse
from collections import Counter
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=str)
    parser.add_argument("--text", required=True, type=str)
    parser.add_argument("--lines", type=int, required=False, default=-1)
    parser.add_argument("--seed", type=int, required=False, default=666)

    return parser.parse_args()


def sample_text(fn, n_lines):
    with open(fn) as fin:
        file_size = sum(1 for _ in tqdm(fin))
    if n_lines > 0:
        percentage = n_lines / file_size
    else:
        percentage = 1.0
    with open(fn) as fin:
        for line in tqdm(fin, total=file_size):
            if np.random.rand() <= percentage:
                yield line.strip()


def measure_usage(tokenizer, text):
    used_tokens = Counter()
    doc_lengths_ws = []
    doc_lengths_tok = []
    for doc in text:
        doc_tok = tokenizer.tokenize(doc)
        doc_lengths_tok.append(len(doc_tok))
        doc_lengths_ws.append(len(doc.split()))
        used_tokens.update(doc_tok)
    return used_tokens, doc_lengths_ws, doc_lengths_tok


def main():
    args = get_args()

    np.random.seed(args.seed)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)

    used_tokens, doc_lengths_ws, doc_lengths_tok = measure_usage(tokenizer, sample_text(args.text, args.lines))

    total_toks = sum(v for v in used_tokens.values())
    unique_toks = len(used_tokens)
    avg_usage_tok = np.mean([v for v in used_tokens.values()])
    median_usage_tok = np.median([v for v in used_tokens.values()])
    std_usage_tok = np.std([v for v in used_tokens.values()])
    avg_doc_length_tok = np.mean(doc_lengths_tok)
    avg_doc_length_ws = np.mean(doc_lengths_ws)
    fertilities = np.array([t / w for w, t in zip(doc_lengths_ws, doc_lengths_tok)])
    fertility = np.mean(fertilities)

    print(f"Total tokens: {total_toks}")
    print(f"Unique tokens: {unique_toks}")
    print(f"Average token usage: {avg_usage_tok}")
    print(f"Median token usage: {median_usage_tok}")
    print(f"Standard deviation token usage: {std_usage_tok}")
    print(f"Average document length whitespace: {avg_doc_length_ws}")
    print(f"Average document length tokenized: {avg_doc_length_tok}")
    print(f"Average fertility: {fertility}")


if __name__ == "__main__":
    main()
