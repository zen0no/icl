# code inspired https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/dt.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(AbsolutePositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, timesteps: torch.FloatTensor):
        _2i = torch.arange(0, self.hidden_dim, step=2, device=timesteps.device).float()
    
        timesteps = timesteps.unsqueeze(-1)
        timesteps = timesteps.expand(timesteps.shape[:-1] + _2i.shape)
        even = torch.sin(timesteps / (10000 ** (_2i / self.hidden_dim)))
        odd = torch.cos(timesteps / (10000 ** (_2i / self.hidden_dim)))

        timesteps_embeddings = torch.stack([even, odd], axis=-1).reshape(timesteps.shape[:-1] + (self.hidden_dim,))
        
        return timesteps_embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, feedforward_dim: int, num_heads: int=4, attn_dropout: float=0., res_dropout: float=0.):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=attn_dropout,
                                          batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
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

        attn_out = self.attn(query=x,
                             key=x,
                             value=x,
                             attn_mask=causal_mask,
                             key_padding_mask=padding_mask,
                             need_weights=False)[0]

        x = self.ln1(x + self.res_dropout(attn_out))
        x = self.ln2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, feedforward_dim: int = 2048,
                 seq_len: int = 40, num_blocks: int = 4, num_attention_heads: int = 4,
                 attn_dropout: float =0., res_dropout: float =0., embed_dropout: float = 0.):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim

        self.time_embedding = AbsolutePositionalEncoding(hidden_dim)

        self.state_embedding = nn.Embedding(state_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.reward_embedding = nn.Linear(1, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                seq_len=3 * seq_len,
                feedforward_dim=feedforward_dim,
                num_heads=num_attention_heads,
                attn_dropout=attn_dropout,
                res_dropout=res_dropout
            )
            for _ in range(num_blocks)
        ])

        self.action_head = nn.Sequential(
                                nn.Linear(hidden_dim, action_dim)
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
            mask: torch.Tensor = None
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # absolute positional encoding
        timestep_embeds = self.time_embedding(timesteps)

        states_embeds = self.state_embedding(states) + timestep_embeds
        actions_embeds = self.action_embedding(actions) + timestep_embeds
        rewards_embeds = self.reward_embedding(rewards) + timestep_embeds

        sequence = (
            torch.stack([states_embeds, actions_embeds, rewards_embeds], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.hidden_dim)
        )

        if mask is not None:
            mask = (
                torch.stack([mask, mask, mask], axis=-1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.embed_ln(self.embed_dropout(sequence))

        for b in self.transformer_blocks:
            out = b(out, padding_mask=mask)
            out[torch.isnan(out)] = 0

        out = self.action_head(out[:, 0::3])

        return out
