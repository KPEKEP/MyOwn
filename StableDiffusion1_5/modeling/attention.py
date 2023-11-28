import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True) -> None:
        """
        Self-Attention Layer.

        Parameters:
        - n_heads (int): Number of attention heads.
        - d_embed (int): Dimension of the input embeddings.
        - in_proj_bias (bool): Whether to use bias in input projection.
        - out_proj_bias (bool): Whether to use bias in output projection.
        """
        super().__init__()

        # Combine the Wq, Wk, and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # Represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: Tensor, causal_mask: bool = False) -> Tensor:
        """
        Forward pass for Self-Attention layer.

        Parameters:
        - x (Tensor): Input tensor of shape (Batch_Size, Seq_Len, Dim).
        - causal_mask (bool): Whether to use a causal mask.

        Returns:
        - Tensor: Output tensor of shape (Batch_Size, Seq_Len, Dim).
        """
        # Initial shape: (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        # Reshape dimensions for multi-head attention
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Input projection: (Batch_Size, Seq_Len, Dim * 3)
        # Split into query, key, value: Each of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Reshape for multi-head attention and transpose
        # Shape: (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Weight calculation: (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        # Apply causal mask if specified
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        # Scale the weights
        weight /= math.sqrt(self.d_head)

        # Softmax normalization
        weight = F.softmax(weight, dim=-1)

        # Weighted sum to produce output
        # Shape: (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # Reshape and transpose to combine heads
        # Shape: (Batch_Size, Seq_Len, Dim)
        output = output.transpose(1, 2).contiguous().view(input_shape)

        # Output projection
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias: bool = True,
                 out_proj_bias: bool = True) -> None:
        """
        Cross-Attention Layer.

        Parameters:
        - n_heads (int): Number of attention heads.
        - d_embed (int): Dimension of the query embeddings.
        - d_cross (int): Dimension of the key/value embeddings.
        - in_proj_bias (bool): Whether to use bias in input projection.
        - out_proj_bias (bool): Whether to use bias in output projection.
        """
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass for Cross-Attention layer.

        Parameters:
        - x (Tensor): Query tensor of shape (Batch_Size, Seq_Len_Q, Dim_Q).
        - y (Tensor): Context tensor of shape (Batch_Size, Seq_Len_KV, Dim_KV).

        Returns:
        - Tensor: Output tensor of shape (Batch_Size, Seq_Len_Q, Dim_Q).
        """
        # Initial shape: (Batch_Size, Seq_Len_Q, Dim_Q)
        input_shape = x.shape

        batch_size, _, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Input projections
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Reshape for multi-head attention and transpose
        # Shape: (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Weight calculation: (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)

        # Scale the weights
        weight /= math.sqrt(self.d_head)

        # Softmax normalization
        weight = F.softmax(weight, dim=-1)

        # Weighted sum to produce output
        # Shape: (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # Reshape and transpose to combine heads
        # Shape: (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.transpose(1, 2).contiguous().view(input_shape)

        # Output projection
        output = self.out_proj(output)

        return output