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
    print("inputs", input[0])
    batch_size, token_len = input.shape

    # token embeddings + position embeddings
    x = (
        wte[input]  # (batch_size, token_len, n_embd)
        + wpe[
            torch.arange(token_len)  # [0, 1, 2, ..., T-1]
        ]  # (token_len, n_embd)
    )  # (batch_size, token_len, n_embd)

    print("x", x)

    # mask for causal attention
    mask = torch.tril(torch.ones(token_len, token_len))

    # residual stream
    for i, block in enumerate(blocks):
        ########################################
        # layer norm 1
        ########################################
        ln_1 = F.layer_norm(
            x, (n_embd,), weight=block["ln_1"]["weight"], bias=block["ln_1"]["bias"]
        )  # (batch_size, token_len, n_embd)

        ########################################
        # multi-head self-attention
        ########################################
        heads = block["attn"]

        # query, key, value
        qkv = (
            ln_1 @ heads["c_attn_weight"] + heads["c_attn_bias"]
        )  # (batch_size, token_len, n_embd * 3)
        if i == 0:
            print("qkv", qkv)
            print("qkv.shape", qkv.shape)
        q, k, v = qkv.split(n_embd, dim=-1)  # (batch_size, token_len, n_embd)

        if i == 0:
            print("q", q)

        # Reshape q, k, v to separate the heads
        q = q.view(batch_size, token_len, n_head, d_head).transpose(
            1, 2
        )  # (batch_size, n_head, token_len, d_head)
        k = k.view(batch_size, token_len, n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, token_len, n_head, d_head).transpose(1, 2)
        print("q_new.shape", q.shape)

        # attention scores + mask
        attn: torch.Tensor = (
            q @ k.transpose(-2, -1)
        ) * d_head**-0.5  # (batch_size, n_head, token_len, token_len)
        print("attn.shape", attn.shape)
        attn = attn.masked_fill(mask[None, None, :, :] == 0, float("-inf"))
        scores = F.softmax(attn, dim=-1)

        heads_output = scores @ v  # (batch_size, n_head, token_len, d_head)

        # merge heads + linear layer
        concat_heads = (
            heads_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, token_len, n_embd)
        )  # (batch_size, token_len, n_embd)
        attn_output = (
            concat_heads @ heads["c_proj_weight"] + heads["c_proj_bias"]
        )  # (batch_size, token_len, n_embd)

        ########################################
        # residual connection 1 + layer norm 2
        ########################################
        x = x + attn_output  # (batch_size, token_len, n_embd)
        print("POST MHA", x.shape)
        print("POST MHA", x)
        ln_2 = F.layer_norm(
            x, (n_embd,), weight=block["ln_2"]["weight"], bias=block["ln_2"]["bias"]
        )

        ########################################
        # mlp
        ########################################
        mlp = block["mlp"]

        # hidden layer
        mlp_output = (
            ln_2 @ mlp["c_fc_weight"] + mlp["c_fc_bias"]
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

        if i == 0:
            print("block", x)

    # final layer norm
    x = F.layer_norm(x, (n_embd,), weight=ln_f["weight"], bias=ln_f["bias"])

    # Project back to vocabulary
    logits = x @ wte.T

    return logits


def generate(input: str) -> str:
    tokens = tokenizer(input, return_tensors="pt").input_ids  # (1, token_len)
    # Generate up to 50 new tokens
    for _ in range(50):
        with torch.no_grad():
            logits = forward(tokens)
        # Convert logits to probabilities
        probs = F.softmax(logits[0, -1, :], dim=-1)
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        # Stop if we generate an EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


print(generate("Not all heroes wear"))
