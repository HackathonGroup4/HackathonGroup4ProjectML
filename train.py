from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import argparse
import os
import sys
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    df = pd.read_csv(args.training_dir)
    train, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["BaseCommand"]
    )
    train_texts = train.to_numpy()[:, 0].tolist()
    train_labels = train.to_numpy()[:, 2].tolist()
    test_texts = test.to_numpy()[:, 0].tolist()
    test_labels = test.to_numpy()[:, 2].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = Dataset(train_encodings, train_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        eval_dataset=test_dataset,
    )
    logger.info("Start training...")
    trainer.train()
    logger.info("Stop training...")

    logger.info("Saving model...")
    trainer.save_model(args.model_dir)
