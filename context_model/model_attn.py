"""
Attention-based Context Model: Permutation-invariant demonstration aggregation.

Replaces the recurrent GRUCell with a self-attention / attention pooling mechanism.
Because transformations like recolor (color permutations) are set-based rather than
sequential, attention removes the artificial sequence-order bias of recurrent states.

Architecture
------------
1. Demo Pair Encoder (MLP):
   demo_pair (150) → Linear(256) → ReLU → Linear(state_dim)

2. Permutation-Invariant Set Aggregator:
   - Stack of encoded demos: (B, D, state_dim)
   - Transformer Encoder (Self-Attention, no positional encoding)
   - Learnable Attention Pooling: computes normalized weights over demos → (B, state_dim)

3. Predictor (MLP):
   concat(aggregated_state, test_input_75) → Linear(256) → ReLU → Linear(75) → (B, 25, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

from data_utils import GRID_FEATURES, GRID_CELLS, NUM_COLORS

DEMO_PAIR_DIM = GRID_FEATURES * 2  # 150
TEST_INPUT_DIM = GRID_FEATURES      # 75
OUTPUT_DIM = GRID_CELLS * NUM_COLORS  # 75


class ContextModelAttn(nn.Module):
    """Attention-based context model for permutation-invariant demo aggregation.

    Parameters
    ----------
    state_dim : int
        Dimension of latent state / attention feature size (default: 128).
    hidden_dim : int
        Hidden dimension of encoder and predictor MLPs (default: 256).
    num_heads : int
        Number of attention heads (default: 4).
    num_layers : int
        Number of Transformer encoder layers (default: 1).
    dropout : float
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 1. Demo Encoder (encodes each demo pair independently)
        self.demo_encoder = nn.Sequential(
            nn.Linear(DEMO_PAIR_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. Transformer Encoder Layer (Self-Attention over demo set)
        # Note: No positional embeddings are added -> strictly permutation-invariant.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 3. Attention Pooling Layer (learns soft weights over demos)
        self.attn_pool_gate = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.Tanh(),
            nn.Linear(state_dim // 2, 1),
        )

        # 4. Predictor MLP: [state || test_input] -> 25x3 logits
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + TEST_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

    def encode_demos(self, demo_tensor: torch.Tensor) -> torch.Tensor:
        """Encode demo pair tensors.

        Parameters
        ----------
        demo_tensor : Tensor of shape (..., 150)

        Returns
        -------
        encoded : Tensor of shape (..., state_dim)
        """
        return self.demo_encoder(demo_tensor)

    def aggregate_demos(
        self,
        encoded_demos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate a sequence of encoded demos into a fixed-size state vector.

        Parameters
        ----------
        encoded_demos : Tensor (B, D, state_dim)
            Encoded demonstration vectors.
        key_padding_mask : BoolTensor (B, D), optional
            True for padded demo positions that should be ignored in attention.

        Returns
        -------
        state : Tensor (B, state_dim)
            Aggregated state representation.
        """
        B, D, d = encoded_demos.shape
        if D == 0:
            return torch.zeros(B, self.state_dim, device=encoded_demos.device)

        # Self-attention over demonstrations
        attn_out = self.transformer_encoder(
            encoded_demos,
            src_key_padding_mask=key_padding_mask,
        )  # (B, D, state_dim)

        # Attention pooling scores
        raw_scores = self.attn_pool_gate(attn_out).squeeze(-1)  # (B, D)
        if key_padding_mask is not None:
            raw_scores = raw_scores.masked_fill(key_padding_mask, -1e9)

        attn_weights = F.softmax(raw_scores, dim=-1).unsqueeze(-1)  # (B, D, 1)

        # Weighted sum of demo representations
        pooled_state = torch.sum(attn_out * attn_weights, dim=1)  # (B, state_dim)
        return pooled_state

    def predict(self, state: torch.Tensor, test_input: torch.Tensor) -> torch.Tensor:
        """Predict output grid logits from aggregated state and test input.

        Parameters
        ----------
        state : Tensor (B, state_dim)
        test_input : Tensor (B, 75)

        Returns
        -------
        logits : Tensor (B, 25, 3)
        """
        combined = torch.cat([state, test_input], dim=-1)
        raw = self.predictor(combined)  # (B, 75)
        return raw.view(-1, GRID_CELLS, NUM_COLORS)

    def forward(
        self,
        demo_pairs_list: List[torch.Tensor],
        test_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of puzzles.

        Parameters
        ----------
        demo_pairs_list : list of Tensor
            List of length B, each of shape (D_i, 150).
        test_inputs : Tensor (B, 75)

        Returns
        -------
        logits : Tensor (B, 25, 3)
        final_states : Tensor (B, state_dim)
        """
        batch_size = len(demo_pairs_list)
        device = test_inputs.device

        max_demos = max(d.shape[0] for d in demo_pairs_list) if demo_pairs_list else 0
        if max_demos == 0:
            state = torch.zeros(batch_size, self.state_dim, device=device)
            logits = self.predict(state, test_inputs)
            return logits, state

        # Create padded batch of demos
        padded_demos = torch.zeros(batch_size, max_demos, DEMO_PAIR_DIM, device=device)
        key_padding_mask = torch.ones(batch_size, max_demos, dtype=torch.bool, device=device)

        for i, demos in enumerate(demo_pairs_list):
            d_len = demos.shape[0]
            if d_len > 0:
                padded_demos[i, :d_len] = demos
                key_padding_mask[i, :d_len] = False

        # Encode and aggregate
        encoded_demos = self.encode_demos(padded_demos)  # (B, max_demos, state_dim)
        state = self.aggregate_demos(encoded_demos, key_padding_mask=key_padding_mask)

        logits = self.predict(state, test_inputs)
        return logits, state

    def forward_with_state_trace(
        self,
        demo_pairs: torch.Tensor,
        test_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Single-instance forward pass returning state after 1, 2, ..., D demos.

        Parameters
        ----------
        demo_pairs : Tensor (D, 150)
        test_input : Tensor (75,)

        Returns
        -------
        logits : Tensor (25, 3)
        states : list of Tensor (state_dim,) for each prefix length 1..D
        """
        device = demo_pairs.device
        D = demo_pairs.shape[0]
        states = []

        if D == 0:
            state = torch.zeros(self.state_dim, device=device)
            logits = self.predict(state.unsqueeze(0), test_input.unsqueeze(0)).squeeze(0)
            return logits, [state]

        # Compute progressive aggregation for sub-slices 1..i
        for i in range(1, D + 1):
            sub_demos = demo_pairs[:i].unsqueeze(0)  # (1, i, 150)
            sub_enc = self.encode_demos(sub_demos)   # (1, i, state_dim)
            sub_state = self.aggregate_demos(sub_enc).squeeze(0)  # (state_dim,)
            states.append(sub_state.detach().clone())

        final_state = states[-1].unsqueeze(0)  # (1, state_dim)
        logits = self.predict(final_state, test_input.unsqueeze(0)).squeeze(0)  # (25, 3)
        return logits, states
