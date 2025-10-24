from pathlib import Path

import click

import torch
from safetensors.torch import load_file

from create_model import create_model


@click.command()
@click.argument("input_text")
def main(input_text):
    model, tokenizer = create_model()
    checkpoint_tensors = Path(__file__).resolve().parent / "checkpoints/best/model.safetensors"
    state = load_file(checkpoint_tensors)
    model.load_state_dict(state)

    input_tokens = tokenizer.encode(input_text)
    input_batch = torch.tensor(input_tokens).unsqueeze(0)
    input_batch = input_batch.to(next(model.parameters()).device)
    output_logits = model(input_batch).logits[0, -1, :]
    output_probs = torch.nn.functional.softmax(output_logits, dim=-1).tolist()
    ham_prob, spam_prob = output_probs
    print(f"Ham: {ham_prob * 100:.2f} || Spam: {spam_prob * 100:.2f}")


if __name__ == "__main__":
    main()
