import torch
from torch import nn
from torch.nn import functional as F

class SpanEmbedding(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            embedding_size: int,
            dropout: float,
            use_head_attn: bool = False,
            return_width_embedding: bool = False):
        self.use_head_attn = use_head_attn
        self.self_attn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        nn.init.xavier_uniform_(self.self_attn[1].weight)
        nn.init.uniform_(self.self_attn[1].bias)
        nn.init.xavier_uniform_(self.self_attn[3].weight)
        nn.init.uniform_(self.self_attn[3].bias)
        self.return_width_embedding = return_width_embedding
        if self.return_width_embedding:
            self.width_embedding = nn.Embedding(5, embedding_size)

    def forward(self, x, continuous_embeddings, width):
        # x: [batch_size, seq_len, input_size]
        # context: [batch_size, seq_len, input_size]
        # width: [batch_size, seq_len]
        # output: [batch_size, seq_len, embedding_size]
        v = x
        if self.use_head_attn:
            max_length = max(len(v) for v in continuous_embeddings)
            padded_tokens_embeddings = torch.stack(
                [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
                for emb in continuous_embeddings]
            )
            masks = torch.stack(
                [torch.cat(
                    (torch.ones(len(emb), device=self.device),
                    torch.zeros(max_length - len(emb), device=self.device)))
                for emb in continuous_embeddings]
            )
            attn_scores = self.self_attn(padded_tokens_embeddings).squeeze(-1)
            attn_scores = attn_scores.masked_fill(masks == 0, -1e9)
            attn_scores = F.softmax(attn_scores, dim=-1)
            v = torch.bmm(attn_scores.unsqueeze(1), padded_tokens_embeddings).squeeze(1)

        if self.return_width_embedding:
            width_embedding = self.width_embedding(torch.clamp(width, 0, 4))
            v = torch.cat((v, width_embedding), dim=-1)

        return v

