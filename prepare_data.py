import sys
import os
from itertools import chain
from typing import List, Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizerFast
from get_args import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments
# TODO
# do all the data pre-processing independent of the training
# save the final result as a new dataset with save_to_disk
# adjust the used tokenize_function & group_texts


def load_doc_per_line_data(path):
    dataset = load_dataset("text", data_files={"_": path})
    dataset = dataset["_"]
    return dataset


def load_tokenizer(path, unk, mask, pad, bos, eos):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    tokenizer.bos_token = bos
    tokenizer.eos_token = eos
    tokenizer.cls_token = bos
    tokenizer.sep_token = eos
    tokenizer.mask_token = mask
    tokenizer.unk_token = unk
    tokenizer.pad_token = pad
    return tokenizer


def tokenize_function(examples,
                      tokenizer,
                      text_column_name,
                      padding,
                      max_seq_length: Optional[int] = None):
    return tokenizer(
        [
            line for line in examples[text_column_name]
            if len(line) > 0 and not line.isspace()
        ],
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )


# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples, max_seq_length):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k]))
        for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [
            t[i:i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def train_val_split(dataset: Dataset,
                    validation_split_percentage: float) -> DatasetDict:
    dataset = dataset.shuffle()
    val_test_size = int(dataset.num_rows * validation_split_percentage)
    val = dataset.select(range(0, val_test_size, 1))
    test = dataset.select(range(len(dataset) - val_test_size, len(dataset), 1))
    train = dataset.select(
        range(val_test_size, dataset.num_rows - val_test_size, 1))
    new_dataset = DatasetDict({
        "train": train,
        "validation": val,
        "test": test
    })
    return new_dataset


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    column_names = ["text"]
    with training_args.main_process_first(desc="load dataset"):
        raw_dataset = load_doc_per_line_data(data_args.train_file)

    with training_args.main_process_first(desc="train-val-test split"):
        raw_dataset = train_val_split(raw_dataset,
                                      data_args.validation_split_percentage)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenizer = load_tokenizer(model_args.tokenizer_path,
                                   "[UNK]", "[MASK]", "[PAD]", "[CLS]",
                                   "[SEP]")
        tok_fun = lambda x: tokenize_function(x,
                                              tokenizer,
                                              "text",
                                              padding=True,
                                              max_seq_length=data_args.
                                              max_seq_length)
        tokenized_datasets = raw_dataset.map(
            tok_fun,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        group_fun = lambda x: group_texts(x, data_args.max_seq_length)
        grouped_datasets = tokenized_datasets.map(
            group_fun,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {data_args.max_seq_length}",
        )
        grouped_datasets.save_to_disk(data_args.tokenized_and_grouped_data)


if __name__ == "__main__":
    main()
