# Naive LLama2 implementation for inference
# Pavel Nakaznenko, 2023

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """
    ModelArgs holds the configuration parameters for the transformer model.

    Attributes:
        dim (int): The dimensions of embeddings. Default is 4096.
        n_layers (int): The number of transformer layers. Default is 32.
        n_heads (int): The number of heads of queries. Default is 32.
        n_kv_heads (Optional[int]): The number of heads for Keys and Values. Defaults to None.
        vocab_size (int): The vocabulary size which will be set by a tokenizer upon the load. Default is -1.
        multiple_of (int): A factor for rounding up dimensions. Default is 256.
        ffn_dim_multiplier (Optional[float]): Multiplier for FeedForward network hidden layer size. Defaults to None.
        norm_eps (float): A small number to prevent division by zero in normalization. Default is 1e-5.
        max_batch_size (int): Maximum batch size for KV cache. Default is 32.
        max_seq_len (int): Maximum sequence length. Default is 2048.
        device (str): Device for torch tensors. Default is None.
    """

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    """Calculates frequencies using theta and position parameters for rotary embeddings."""
    assert head_dim % 2 == 0, "Dims must be dividable by two"
    theta_power_num = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_power_num / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)  # Position parameter
    freqs = torch.outer(m, theta).float()  # Frequency for each position
    return torch.polar(torch.ones_like(freqs), freqs)  # Complex numbers in polar form


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    """Applies rotary embeddings to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable weight for scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization to the input tensor."""
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class SelfAttention(nn.Module):
    """Self Attention Block."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # Query linear transformation
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Key linear transformation
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Value linear transformation
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # Output linear transformation

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))  # Key cache
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))  # Value cache

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats KV heads to match the number of query heads."""
        if n_rep == 1:
            return x
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len,
                                                                                                 n_kv_heads * n_rep,
                                                                                                 head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        """Computes the self-attention and updates the KV cache."""
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk, xv = [y.view(*x.shape[:2], self.n_heads_q, self.head_dim) for y in [xq, xk, xv]]
        xq, xk = [apply_rotary_embeddings(y, freqs_complex, device=x.device) for y in [xq, xk]]

        self.cache_k[:x.size(0), start_pos:start_pos + x.size(1)] = xk
        self.cache_v[:x.size(0), start_pos:start_pos + x.size(1)] = xv
        keys, values = self._repeat_kv(self.cache_k[:x.size(0), :start_pos + x.size(1)], self.n_rep), self._repeat_kv(
            self.cache_v[:x.size(0), :start_pos + x.size(1)], self.n_rep)

        scores = torch.matmul(xq.transpose(1, 2), keys.transpose(1, 2).transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values.transpose(1, 2)).transpose(1, 2).contiguous().view(*x.shape[:2], -1)
        return self.wo(output)  # Output linear transformation


class FeedForward(nn.Module):
    """Feed Forward Network Block."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)  # First linear transformation
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # Second linear transformation
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # Third linear transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Feed Forward Network to the input tensor."""
        swish = F.silu(self.w1(x))
        xv = self.w3(x)
        x = swish * xv
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder Block consisting of a self-attention and feed-forward network."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)  # Self Attention block
        self.feed_forward = FeedForward(args)  # Feed Forward Network

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # Normalization before self-attention
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # Normalization before feed-forward network

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        """Processes the input tensor through the encoder block."""
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Transformer consisting of a sequence of encoder blocks."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)  # Token Embeddings

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))  # Adding Encoder Blocks

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)  # Final normalization
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)  # Output Linear Transformation

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
            device=self.args.device)  # Precomputed frequencies for rotary embeddings

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Processes the input tokens through the transformer."""
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        h = self.tok_embeddings(tokens)  # Token embeddings

        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]  # Frequencies for rotary embeddings

        for layer in self.layers:  # Processing through each encoder block
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)  # Final normalization
        output = self.output(h).float()  # Output Linear Transformation
        return output
