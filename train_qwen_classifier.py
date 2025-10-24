import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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



def main():
    model_name = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )
    model.lm_head = torch.nn.Linear(
        in_features=model.lm_head.in_features,
        out_features=2,
        device=model.lm_head.weight.device,
        dtype=model.lm_head.weight.dtype,
    )

    train_loader, val_loader, test_loader = load_datasets(tokenizer)
    train(model, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
