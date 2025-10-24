import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def create_model():
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

    return model, tokenizer
