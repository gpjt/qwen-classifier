import datetime
import json
import time
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file

from torch.utils.data import DataLoader, Dataset

from create_model import create_model


class SpamDataset(Dataset):

    def __init__(
        self, csv_file, tokenizer, max_length=None, pad_token_id=50256
    ):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]


    def __getitem__(self, ix):
        encoded = self.encoded_texts[ix]
        label = self.data.iloc[ix]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


    def __len__(self):
        return len(self.data)


    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length



def load_datasets(tokenizer):
    train_dataset = SpamDataset(
        csv_file="classification-train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file="classification-validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="classification-test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def calc_accuracy_loader(data_loader, model, num_batches=None):
    device = next(model.parameters()).device
    model.eval()
    correct_predictions = 0
    num_examples = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch).logits[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)

        num_examples += predicted_labels.shape[0]
        correct_predictions += (
            (predicted_labels == target_batch).sum().item()
        )

    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model):
    device = next(model.parameters()).device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch).logits[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss



def save_checkpoint(model, epoch, train_loss, val_loss, is_best_val_loss):
    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    if checkpoints_dir.exists():
        assert checkpoints_dir.is_dir()
    else:
        checkpoints_dir.mkdir()

    now = datetime.datetime.now(datetime.UTC)
    checkpoint_dir = checkpoints_dir / f"{now:%Y%m%dZ%H%M%S}"
    checkpoint_dir.mkdir()

    tensors_file = checkpoint_dir / "model.safetensors"
    save_file(model.state_dict(), tensors_file)

    meta = dict(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
    )
    meta_file = checkpoint_dir / "meta.json"
    meta_file.write_text(json.dumps(meta))

    symlink_target = Path(".") / checkpoint_dir.name
    if is_best_val_loss:
        best_path = checkpoints_dir / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)


def train_classifier_simple(
    model, train_loader, val_loader, optimizer,
    num_epochs, eval_freq, eval_iter
):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    examples_seen = 0
    global_step = -1

    best_val_loss = None

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f} "
                    f"Val loss {val_loss:.3f}"
                )
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best_val_loss = True
                else:
                    is_best_val_loss = False
                save_checkpoint(model, epoch, train_loss, val_loss, is_best_val_loss)

        train_accuracy = calc_accuracy_loader(
            train_loader, model, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def train(model, train_loader, val_loader, test_loader):
    train_accuracy = calc_accuracy_loader(
        train_loader, model, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, num_batches=10
    )

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, num_batches=5)
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_classifier_simple(
        model, train_loader, val_loader, optimizer,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")



def main():
    model, tokenizer = create_model()

    train_loader, val_loader, test_loader = load_datasets(tokenizer)
    train(model, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
