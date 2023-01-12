import argparse
from transformers import PreTrainedTokenizerFast
from tokenizers.decoders import ByteLevel
from typing import Set, Iterable, Tuple
from pprint import pprint

bpe_char = "Ġ"
meta_char = "▁"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    tok_types = ["byte", "meta", "wp"]
    parser.add_argument("--tok-a", type=str)
    parser.add_argument("--type-a", type=str, choices=tok_types)
    parser.add_argument("--tok-b", type=str)
    parser.add_argument("--type-b", type=str, choices=tok_types)

    return parser.parse_args()


def extract_vocab(tokenizer: PreTrainedTokenizerFast,
                  tok_type: str
                  ) -> Set[str]:
    decoder = ByteLevel()
    vocab = set()
    double_cnt = 0
    for word in tokenizer.vocab.keys():
        if tok_type == "byte":
            if word.startswith(bpe_char):
                vocab.add(meta_char + decoder.decode([word])[1:])
            else:
                vocab.add(decoder.decode([word]))
        elif tok_type == "wp":
            if word.startswith("##"):
                if word[2:] in vocab and len(word[2:]) > 1:
                    double_cnt += 1
                    print("prefix", word)
                # vocab.add(word[2:])
                vocab.add(word)
            else:
                if len(word) > 1:
                    if word in vocab:
                        double_cnt += 1
                        print("no prefix gt 1", word)
                    vocab.add(meta_char + word)
                else:
                    if word in vocab:
                        double_cnt += 1
                        print("no prefix eq 1", word)
                    vocab.add(word)
        elif tok_type == "meta":
            vocab.add(word)
    print("doubles: ", double_cnt)
    return vocab


def prefix_suffix(words: Iterable[str], tok_type: str) -> Tuple[Iterable[str], Iterable[str]]:
    prefixes = set()
    suffixes = set()
    singles = set()
    for word in words:
        if len(word) == 1:
            singles.add(word)
        elif tok_type in ["byte", "meta"]:
            if word.startswith(meta_char) or word.startswith(bpe_char):
                prefixes.add(word)
            else:
                suffixes.add(word)
        elif tok_type == "wp":
            if word.startswith("##"):
                suffixes.add(word)
            else:
                prefixes.add(word)
    return singles, prefixes, suffixes


def main():
    args = get_args()

    tokenizer_a = PreTrainedTokenizerFast(tokenizer_file=args.tok_a)
    tokenizer_b = PreTrainedTokenizerFast(tokenizer_file=args.tok_b)

    if args.type_a == args.type_b:
        voc_a = extract_vocab(tokenizer_a, "meta")
        voc_b = extract_vocab(tokenizer_b, "meta")
        tok_type = args.type_a
    else:
        voc_a = extract_vocab(tokenizer_a, args.type_a)
        voc_b = extract_vocab(tokenizer_b, args.type_b)
        tok_type = "meta" 

    print(len(voc_a), len(voc_b))
    print(len(voc_a.union(voc_b)))
    print(len(voc_a.intersection(voc_b)))
    print(len(voc_a.difference(voc_b)))
    print(len(voc_b.difference(voc_a)))

    o, p, s = prefix_suffix(voc_a, tok_type)
    print(len(o), len(p), len(s), (len(p) + len(s)))
    o, p, s = prefix_suffix(voc_b, tok_type)
    print(len(o), len(p), len(s), (len(p) + len(s)))
    o, p, s = prefix_suffix(voc_a.difference(voc_b), tok_type)
    print(len(o), len(p), len(s), (len(p) + len(s)))
    o, p, s = prefix_suffix(voc_b.difference(voc_a), tok_type)
    print(len(o), len(p), len(s), (len(p) + len(s)))
    print(p)

    # pprint((voc_a.difference(voc_b)))
    # pprint((voc_b.difference(voc_a)))


if __name__ == "__main__":
    main()
