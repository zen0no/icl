# code inspired https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/dt.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, num_heads: int=4, attn_dropout: float=0., res_dropout: float=0.):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=attn_dropout,
                                          batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.res_dropout = nn.Dropout(res_dropout)
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    def forward(
            self,
            x: torch.Tensor,
            padding_mask: torch.Tensor = None
    ) -> torch.FloatTensor:
        causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]]

        norm_x = self.ln1(x)
        attn_out = self.attn(query=norm_x,
                             key=norm_x,
                             value=norm_x,
                             attn_mask=causal_mask,
                             key_padding_mask=padding_mask,
                             need_weights=False)[0]

        x = x + self.res_dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, seq_len=40, num_blocks=4, num_attention_heads=4,
                 attn_dropout=0., res_dropout=0., embed_dropout=0.):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim

        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.reward_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(seq_len, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                seq_len=3 * seq_len,
                num_heads=num_attention_heads,
                attn_dropout=attn_dropout,
                res_dropout=res_dropout
            )
            for _ in range(num_blocks)
        ])

        self.action_head = nn.Sequential(
                                nn.Linear(hidden_dim, action_dim),
                                nn.Softmax(dim=-1)
        )

        self.embed_dropout = nn.Dropout(embed_dropout)
        self.embed_ln = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            timesteps: torch.Tensor,
            padding_mask: torch.Tensor = None
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]

        timestep_embeds = self.timestep_embedding(timesteps)
        states_embeds = self.state_embedding(states) + timestep_embeds
        actions_embeds = self.action_embedding(actions) + timestep_embeds
        rewards_embeds = self.reward_embedding(rewards) + timestep_embeds

        sequence = (
            torch.stack([states_embeds, actions_embeds, rewards_embeds], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.hidden_dim)
        )

        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.embed_ln(self.embed_dropout(sequence))

        for b in self.transformer_blocks:
            out = b(out, padding_mask=padding_mask)

        out = self.action_head(out[:, 0::3])

        return out
