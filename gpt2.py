import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model, AutoTokenizer
from typing import TypedDict

MODEL_NAME = "gpt2"

model = GPT2Model.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, clean_up_tokenization_spaces=True)

# Hyperparameters
cfg = model.config.to_dict()
vocab_size: int = cfg["vocab_size"]  # 50257, token dictionary size
n_positions: int = cfg["n_positions"]  # 1024, "context length"
n_embd: int = cfg["n_embd"]  # 768, token embedding dimensions
n_head: int = cfg["n_head"]  # 12, attention heads per block
n_layer: int = cfg["n_layer"]  # 12, number of transformer blocks
d_mlp: int = n_embd * 4  # 3072, MLP hidden layer dimensions
d_head: int = n_embd // n_head  # 64, dimensions of each attention head


# Parameters
class Attention(TypedDict):
    c_attn_weight: torch.Tensor  # (768 = 64 * 12, 3072 = 768 * 3)
    c_attn_bias: torch.Tensor  # (768 * 3,)
    c_proj_weight: torch.Tensor  # (768, 768)
    c_proj_bias: torch.Tensor  # (768,)


class MLP(TypedDict):
    c_fc_weight: torch.Tensor  # (768, 3072)
    c_fc_bias: torch.Tensor  # (3072,)
    c_proj_weight: torch.Tensor  # (3072, 768)
    c_proj_bias: torch.Tensor  # (768,)


class LayerNorm(TypedDict):
    weight: torch.Tensor  # (768,)
    bias: torch.Tensor  # (768,)


class Block(TypedDict):
    ln_1: LayerNorm  # layer norm 1
    attn: Attention  # multi-head self-attention
    ln_2: LayerNorm  # layer norm 2
    mlp: MLP  # Feed-forward


state_dict = model.state_dict()
wte: torch.Tensor = state_dict[
    "wte.weight"
]  # (vocab_size, n_embd), word token embeddings
wpe: torch.Tensor = state_dict[
    "wpe.weight"
]  # (n_positions, n_embd), word position embeddings
blocks: list[Block] = []  # transformer blocks
for i in range(n_layer):
    blocks.append(
        {
            "ln_1": {
                "weight": state_dict[f"h.{i}.ln_1.weight"],
                "bias": state_dict[f"h.{i}.ln_1.bias"],
            },
            "attn": {
                "c_attn_weight": state_dict[f"h.{i}.attn.c_attn.weight"],
                "c_attn_bias": state_dict[f"h.{i}.attn.c_attn.bias"],
                "c_proj_weight": state_dict[f"h.{i}.attn.c_proj.weight"],
                "c_proj_bias": state_dict[f"h.{i}.attn.c_proj.bias"],
            },
            "ln_2": {
                "weight": state_dict[f"h.{i}.ln_2.weight"],
                "bias": state_dict[f"h.{i}.ln_2.bias"],
            },
            "mlp": {
                "c_fc_weight": state_dict[f"h.{i}.mlp.c_fc.weight"],
                "c_fc_bias": state_dict[f"h.{i}.mlp.c_fc.bias"],
                "c_proj_weight": state_dict[f"h.{i}.mlp.c_proj.weight"],
                "c_proj_bias": state_dict[f"h.{i}.mlp.c_proj.bias"],
            },
        }
    )
ln_f: LayerNorm = {  # final layer normalization, after the residual stream
    "weight": state_dict["ln_f.weight"],
    "bias": state_dict["ln_f.bias"],
}


def forward(input: torch.Tensor) -> torch.Tensor:
    batch_size, token_len = input.shape

    # token embeddings + position embeddings
    x = (
        wte[input]  # (batch_size, token_len, n_embd)
        + wpe[
            torch.arange(token_len)  # [0, 1, 2, ..., T-1]
        ]  # (token_len, n_embd)
    )  # (batch_size, token_len, n_embd)

    # mask for causal attention
    mask = torch.tril(torch.ones(token_len, token_len))

    # residual stream
    for block in blocks:
        ########################################
        # layer norm 1
        ########################################
        x = F.layer_norm(
            x, (n_embd,), **block["ln_1"]
        )  # (batch_size, token_len, n_embd)

        ########################################
        # multi-head self-attention
        ########################################
        heads = block["attn"]

        # query, key, value
        qkv = (
            x @ heads["c_attn_weight"] + heads["c_attn_bias"]
        )  # (batch_size, token_len, 3 * n_embd)

        # separate heads
        q = (
            qkv[:, :, :n_embd].view(-1, token_len, n_head, d_head).transpose(1, 2)
        )  # (batch_size, n_head, token_len, d_head)
        k = (
            qkv[:, :, n_embd : 2 * n_embd]
            .view(-1, token_len, n_head, d_head)
            .transpose(1, 2)
        )
        v = qkv[:, :, 2 * n_embd :].view(-1, token_len, n_head, d_head).transpose(1, 2)

        # attention scores + mask
        attn: torch.Tensor = (q @ k.transpose(-2, -1)) * (
            1.0 / np.sqrt(d_head)
        )  # (batch_size, n_head, token_len, token_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(attn, dim=-1)

        heads_output = scores @ v  # (batch_size, n_head, token_len, d_head)

        # merge heads + linear layer
        concat_heads = heads_output.view(
            -1, token_len, n_embd
        )  # (batch_size, token_len, n_embd)
        attn_output = (
            concat_heads @ heads["c_proj_weight"] + heads["c_proj_bias"]
        )  # (batch_size, token_len, n_embd)

        ########################################
        # residual connection 1 + layer norm 2
        ########################################
        x = x + attn_output  # (batch_size, token_len, n_embd)
        x = F.layer_norm(x, (n_embd,), **block["ln_2"])

        ########################################
        # mlp
        ########################################
        mlp = block["mlp"]

        # hidden layer
        mlp_output = (
            x @ mlp["c_fc_weight"] + mlp["c_fc_bias"]
        )  # (batch_size, token_len, d_mlp)
        mlp_output = F.gelu(mlp_output)
        # output layer
        mlp_output = (
            mlp_output @ mlp["c_proj_weight"] + mlp["c_proj_bias"]
        )  # (batch_size, token_len, n_embd)

        ########################################
        # residual connection 2
        ########################################
        x = x + mlp_output

    # final layer norm
    x = F.layer_norm(x, (n_embd,), **ln_f)

    # Project back to vocabulary
    logits = x @ wte.T

    return logits


def generate(input: str) -> str:
    tokens = tokenizer(input, return_tensors="pt").input_ids  # (1, token_len)
    # Generate up to 50 new tokens
    for _ in range(50):
        with torch.no_grad():
            logits = forward(tokens)
        next_token = logits[0, -1, :].argmax().unsqueeze(0)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        # Stop if we generate an EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


print(generate("Hello, how are you?"))
