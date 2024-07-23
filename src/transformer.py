import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, embed_dim, expansion_coeff=4, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, expansion_coeff * embed_dim)
        self.fc2 = nn.Linear(expansion_coeff * embed_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0., mlp_dropout=0.):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=attn_dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout=mlp_dropout)

    def forward(self, x):
        pass


class Transformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, seq_len=40, num_blocks=4, num_attention_heads=4, ):
        super(Transformer, self).__init__()
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.reward_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(seq_len, hidden_dim)

    def forward(self, states, actions, rewards):
        pass
