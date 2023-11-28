from typing import Optional

import torch
from torch import nn

from modeling.attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """Embedding layer for CLIP model.

    Attributes:
        token_embedding (nn.Embedding): Token embedding layer.
        position_embedding (nn.Parameter): Positional embedding tensor.
    """

    def __init__(self, n_vocab: int, n_embd: int, n_token: int, position_init: Optional[torch.Tensor] = None):
        """Initialize CLIPEmbedding.

        Parameters:
            n_vocab (int): Vocabulary size.
            n_embd (int): Embedding dimension.
            n_token (int): Number of tokens in a sequence.
            position_init (Optional[torch.Tensor]): Initial tensor for positional embedding.
        """
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(
            position_init if position_init is not None else torch.zeros((n_token, n_embd)))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass for CLIPEmbedding.

        Parameters:
            tokens (torch.LongTensor): Input tokens.

        Returns:
            torch.FloatTensor: Output embeddings.
        """

        # Shape transformation: (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        # Shape transformation: (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    """Single CLIP layer containing self-attention and feed-forward network.

    Attributes:
        layernorm_1 (nn.LayerNorm): Layer normalization before self-attention.
        attention (SelfAttention): Self-attention module.
        layernorm_2 (nn.LayerNorm): Layer normalization before feed-forward network.
        linear_1 (nn.Linear): First linear layer in feed-forward network.
        linear_2 (nn.Linear): Second linear layer in feed-forward network.
    """

    def __init__(self, n_head: int, n_embd: int, ff_hidden_mult: int = 4, gelu_param: float = 1.702):
        """Initialize CLIPLayer.

        Parameters:
            n_head (int): Number of attention heads.
            n_embd (int): Embedding dimension.
            ff_hidden_mult (int): Multiplier for feed-forward hidden layer size.
            gelu_param (float): Parameter for GELU activation.
        """
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, ff_hidden_mult * n_embd)
        self.linear_2 = nn.Linear(ff_hidden_mult * n_embd, n_embd)
        self.gelu_param = gelu_param

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass for CLIPLayer.

        Parameters:
            x (torch.FloatTensor): Input tensor.

        Returns:
            torch.FloatTensor: Output tensor.
        """

        # Shape transformation: (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        # Shape transformation: (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        # QuickGELU activation function
        x = x * torch.sigmoid(self.gelu_param * x)

        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):
    """CLIP model containing embedding layer and multiple CLIP layers.

    Attributes:
        embedding (CLIPEmbedding): Embedding layer.
        layers (nn.ModuleList): List of CLIPLayers.
        layernorm (nn.LayerNorm): Final layer normalization.
    """

    def __init__(self, n_vocab: int = 49408, n_embd: int = 768, n_token: int = 77, n_layers: int = 12,
                 n_head: int = 12):
        """Initialize CLIP.

        Parameters:
            n_vocab (int): Vocabulary size.
            n_embd (int): Embedding dimension.
            n_token (int): Number of tokens in a sequence.
            n_layers (int): Number of CLIPLayers.
            n_head (int): Number of attention heads.
        """
        super().__init__()

        self.embedding = CLIPEmbedding(n_vocab, n_embd, n_token)
        self.layers = nn.ModuleList([CLIPLayer(n_head, n_embd) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(n_embd)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass for CLIP.

        Parameters:
            tokens (torch.LongTensor): Input tokens.

        Returns:
            torch.FloatTensor: Output tensor.
        """

        # Type cast: Ensure tokens are long integers
        tokens = tokens.type(torch.long)

        # Shape transformation: (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers
        for layer in self.layers:
            state = layer(state)

        # Shape transformation: (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output