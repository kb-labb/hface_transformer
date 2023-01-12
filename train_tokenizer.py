import argparse
from typing import Iterable, Dict
from datasets import load_dataset, concatenate_datasets, Dataset
import transformers
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
    Regex,
    SentencePieceBPETokenizer,
)

if True:
    PAD = "[PAD]"
    BOS = "[CLS]"
    EOS = "[SEP]"
    MASK = "[MASK]"
    UNK = "[UNK]"
else:
    PAD = "<pad>"
    BOS = "<s>"
    EOS = "</s>"
    MASK = "<mask>"
    UNK = "<unk>"
# replacement = "▁"
# replacement = None  # default is the above anyways
TOKENIZER_TYPES = ["unigram", "wordpiece", "bpe-straight", "spe-bpe"]


def batch_iterator(dataset: Dataset, dataset_size: int,
                   batch_size: int) -> Iterable[Dict[str, str]]:
    for i in range(0, dataset_size, batch_size):
        yield dataset[i:i + batch_size]["text"]


# https://github.com/huggingface/tokenizers/issues/640#issuecomment-792305076
def tokenizer_trainer(text,
                      tok_type: str,
                      vocab_size: int,
                      tokenizer_file: str = "tokenizer.json",
                      min_frequency: int = 0,
                      add_prefix_space: bool = True,
                      batch_size: int = 50,
                      all_pts: bool = True) -> None:
    # Supply either path to txt file or list of strings as text arg

    assert tok_type in TOKENIZER_TYPES

    # tokenizer = Tokenizer(models.WordPiece(unk_token=UNK))
    if tok_type == "unigram":
        tokenizer = Tokenizer(models.Unigram())

        pts = [
               pre_tokenizers.Metaspace(# replacement=replacement,
                                        add_prefix_space=add_prefix_space),
               pre_tokenizers.Split(Regex("\d"), behavior="merged_with_previous"),
               pre_tokenizers.Punctuation(),
               ]
        if not all_pts:
            pts = [pts[0]]

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pts)

        tokenizer.decoder = decoders.Metaspace()

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, MASK, BOS, EOS],
            min_frequency=min_frequency,
            unk_token=UNK,
            shrinking_factor=0.75,  # 0.75
            max_piece_length=16,  # 16
            n_sub_iterations=2,  # 2
        )

    elif tok_type == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece())

        pts = [
               pre_tokenizers.WhitespaceSplit(),  # does not split on punctuation
               pre_tokenizers.Split(Regex("\d"), behavior="merged_with_previous"),
               pre_tokenizers.Punctuation(),
               ]
        if not all_pts:
            pts = [pts[0]]

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pts)

        tokenizer.decoder = decoders.WordPiece()

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, MASK, BOS, EOS],
            min_frequency=min_frequency,
            unk_token=UNK,
        )

    if tok_type == "bpe-straight":
        tokenizer = Tokenizer(models.BPE())

        pts = [
               pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space),
               pre_tokenizers.Split(Regex("\d"), behavior="merged_with_previous"),
               pre_tokenizers.Punctuation(),
               ]

        if not all_pts:
            pts = [pts[0]]

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pts)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, MASK, BOS, EOS],
            min_frequency=min_frequency,
            # limit_alphabet=1000,
        )

    if tok_type == "spe-bpe":
        tokenizer = Tokenizer(models.BPE())
        pts = [
               pre_tokenizers.Metaspace( # replacement=replacement,
                                        add_prefix_space=add_prefix_space),
               pre_tokenizers.Split(Regex("\d"), behavior="merged_with_previous"),
               pre_tokenizers.Punctuation(),
               ]

        if not all_pts:
            pts = [pts[0]]

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pts)
        tokenizer.decoder = decoders.Metaspace( # replacement=replacement,
                                               add_prefix_space=add_prefix_space)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, MASK, BOS, EOS],
            min_frequency=min_frequency,
            # limit_alphabet=1000,
        )

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Nmt(),
        normalizers.NFKC(),
        # normalizers.NFD(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ])

    if isinstance(text, str):
        # if user specified path to txt file as string
        print("is text")
        tokenizer.train(text, trainer=trainer)
    else:
        # text is a datasets Dataset
        tokenizer.train_from_iterator(batch_iterator(text, len(text),
                                                     batch_size),
                                      trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A {EOS}",
        pair=f"{BOS} $A {EOS} $B:1 {EOS}:1",
        special_tokens=[
            (f"{BOS}", tokenizer.get_vocab()[BOS]),
            (f"{EOS}", tokenizer.get_vocab()[EOS]),
        ],
    )
    tokenizer.save(tokenizer_file, pretty=True)
    # tokenizer.model.save("output_dir")
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
    convert_to_PreTrainedTokenizerFast(tokenizer, None, special_tokens, tokenizer_file)
    return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infiles", nargs="+", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--min_frequency", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--add_prefix_space", action="store_true")
    parser.add_argument("--all_pts", action="store_true")
    parser.add_argument("--tok_type",
                        choices=["bpe-straight", "wordpiece", "unigram", "spe-bpe"],
                        default="wordpiece")

    return parser.parse_args()


def sentencepiece_bpe(text, vocab_size, tokenizer_path, model_length, batch_size):
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
    tk_tokenizer = SentencePieceBPETokenizer()
    tk_tokenizer.train_from_iterator(
            batch_iterator(text, len(text), batch_size),
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens
    )
    tk_tokenizer.save(tokenizer_path)
    return


def convert_to_PreTrainedTokenizerFast(tk_tokenizer, model_length, special_tokens, tokenizer_path):
    # convert
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer,
                                                     model_max_length=model_length,
                                                     unk_token="<unk>",
                                                     pad_token="<pad>",
                                                     bos_token="<s>",
                                                     eos_token="</s>",
                                                     cls_token="<s>",
                                                     sep_token="</s>",
                                                     )
    tokenizer.save_pretrained(tokenizer_path + ".pttf")
    return


if __name__ == "__main__":
    args = get_args()
    dataset = load_dataset(
        "text",
        data_files={str(i): name
                    for i, name in enumerate(args.infiles)},
        cache_dir="cache_dataset",
    )
    dataset = concatenate_datasets(
        [dataset[str(i)] for i, _ in enumerate(args.infiles)])
    tokenizer_trainer(text=dataset,
                      tok_type=args.tok_type,
                      vocab_size=args.vocab_size,
                      tokenizer_file=args.tokenizer_name,
                      min_frequency=args.min_frequency,
                      add_prefix_space=args.add_prefix_space,
                      batch_size=args.batch_size,
                      all_pts=args.all_pts)
