import click

import torch

from persistence import load_model


@click.command()
@click.argument("input_text")
@click.argument("checkpoint", default="best")
def main(input_text, checkpoint):
    model, tokenizer = load_model(checkpoint)

    input_tokens = tokenizer.encode(input_text)
    input_batch = torch.tensor(input_tokens).unsqueeze(0)
    input_batch = input_batch.to(next(model.parameters()).device)

    # Zero'th batch, last token logits
    output_logits = model(input_batch).logits[0, -1, :]

    output_probs = torch.nn.functional.softmax(output_logits, dim=-1)
    ham_prob, spam_prob = output_probs.tolist()
    print(f"Ham: {ham_prob * 100:.2f} || Spam: {spam_prob * 100:.2f}")


if __name__ == "__main__":
    main()
