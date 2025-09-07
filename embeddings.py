import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)        
        self.register_buffer('pe', pe.unsqueeze(0))    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x + self.pe[:, :seq_len, :d_model]
        return self.dropout(x)
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super(LearnedPositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model        
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        return self.dropout(x)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base        
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)        
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            self._cos_cached = freqs.cos().to(dtype)
            self._sin_cached = freqs.sin().to(dtype)    
    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_heads, head_dim = q.shape
        self._update_cos_sin_cache(start_pos + seq_len, q.device, q.dtype)        
        cos = self._cos_cached[start_pos:start_pos + seq_len, :head_dim // 2]
        sin = self._sin_cached[start_pos:start_pos + seq_len, :head_dim // 2]
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)        
        q = q.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)        
        q_rot = self._rotate_half(q, cos, sin)
        k_rot = self._rotate_half(k, cos, sin)        
        q_rot = q_rot.reshape(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        k_rot = k_rot.reshape(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)        
        return q_rot, k_rot    
    def _rotate_half(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
class TechEmbeddingLayer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 max_seq_len: int = 512,
                 dropout: float = 0.1,
                 padding_idx: int = 0,
                 pos_encoding: str = "learned",
                 layer_norm: bool = True):
        super(TechEmbeddingLayer, self).__init__()        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)        
        self.pos_encoding_type = pos_encoding
        if pos_encoding == "sinusoidal":
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        elif pos_encoding == "learned":
            self.pos_encoding = LearnedPositionalEmbedding(max_seq_len, d_model, dropout)
        elif pos_encoding == "rope":
            self.pos_encoding = RotaryPositionalEmbedding(d_model, max_seq_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding}")        
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            nn.init.constant_(self.token_embedding.weight[self.padding_idx], 0.0)    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (input_ids >= self.vocab_size).any():
            raise ValueError(f"Input IDs contain values >= vocab_size ({self.vocab_size})")        
        embeddings = self.token_embedding(input_ids)
        if self.pos_encoding_type != "rope":
            embeddings = self.pos_encoding(embeddings)        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings    
    def get_positional_encoding(self):
        return self.pos_encoding if self.pos_encoding_type == "rope" else None
class AdaptiveEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 cutoffs: list = [2000, 10000],
                 div_val: float = 4.0):
        super(AdaptiveEmbedding, self).__init__()        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.div_val = div_val        
        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()        
        for i in range(len(self.cutoffs) - 1):
            l_idx = self.cutoffs[i]
            r_idx = self.cutoffs[i + 1]
            d_emb = int(d_model / (div_val ** i))            
            emb = nn.Embedding(r_idx - l_idx, d_emb)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self.embeddings.append(emb)            
            if d_emb != d_model:
                proj = nn.Linear(d_emb, d_model, bias=False)
                nn.init.normal_(proj.weight, mean=0.0, std=0.02)
                self.projections.append(proj)
            else:
                self.projections.append(nn.Identity())    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (input_ids >= self.vocab_size).any():
            raise ValueError(f"Input IDs contain values >= vocab_size ({self.vocab_size})")        
        batch_size, seq_len = input_ids.shape
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, 
                               device=input_ids.device, dtype=torch.float32)        
        for i in range(len(self.cutoffs) - 1):
            l_idx = self.cutoffs[i]
            r_idx = self.cutoffs[i + 1]            
            mask = (input_ids >= l_idx) & (input_ids < r_idx)
            if mask.any():
                indices = input_ids[mask] - l_idx
                indices = indices.clamp(max=r_idx - l_idx - 1)
                emb = self.embeddings[i](indices)
                emb = self.projections[i](emb)
                embeddings[mask] = emb        
        return embeddings
def create_padding_mask(input_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    return input_ids == padding_idx
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
def create_attention_mask(input_ids: torch.Tensor, 
                         padding_idx: int = 0,
                         causal: bool = True) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    padding_mask = create_padding_mask(input_ids, padding_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)    
    if causal:
        causal_mask = create_causal_mask(seq_len, device)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        combined_mask = padding_mask | causal_mask
    else:
        combined_mask = padding_mask
    
    return combined_mask
class EmbeddingAnalyzer:
    def __init__(self, embedding_layer: nn.Module):
        self.embedding_layer = embedding_layer    
    def get_similarity_matrix(self, tokens: List[int] = None) -> torch.Tensor:
        if hasattr(self.embedding_layer, 'token_embedding'):
            embeddings = self.embedding_layer.token_embedding.weight
        elif hasattr(self.embedding_layer, 'embeddings'):
            weights = [emb.weight for emb in self.embedding_layer.embeddings]
            embeddings = []
            for i, w in enumerate(weights):
                proj = self.embedding_layer.projections[i]
                embeddings.append(proj(w))
            embeddings = torch.cat(embeddings, dim=0)
        else:
            embeddings = self.embedding_layer.weight        
        if tokens is not None and len(tokens) > 0:
            embeddings = embeddings[tokens]        
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return torch.mm(normalized_embeddings, normalized_embeddings.t())    
    def find_similar_tokens(self, token_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        similarity_matrix = self.get_similarity_matrix()
        similarities = similarity_matrix[token_id]
        top_similarities, top_indices = torch.topk(similarities, top_k + 1)
        mask = top_indices != token_id
        top_similarities = top_similarities[mask][:top_k]
        top_indices = top_indices[mask][:top_k]
        return list(zip(top_indices.tolist(), top_similarities.tolist()))    
    def analyze_embedding_distribution(self):
        if hasattr(self.embedding_layer, 'token_embedding'):
            weights = self.embedding_layer.token_embedding.weight
        elif hasattr(self.embedding_layer, 'embeddings'):
            weights = torch.cat([emb.weight for emb in self.embedding_layer.embeddings], dim=0)
        else:
            weights = self.embedding_layer.weight        
        stats = {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'min': weights.min().item(),
            'max': weights.max().item(),
            'norm_mean': weights.norm(dim=1).mean().item(),
            'norm_std': weights.norm(dim=1).std().item()
        }
        return stats
def test_embeddings():
    print("Testing embedding layers...")    
    vocab_size = 1000
    d_model = 512
    max_seq_len = 128
    batch_size = 4
    seq_len = 64    
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))    
    embedding_types = [
        ("Learned Position", "learned"),
        ("Sinusoidal Position", "sinusoidal"),
        ("RoPE", "rope")
    ]    
    for name, pos_type in embedding_types:
        print(f"\nTesting {name} Embedding:")
        embedding_layer = TechEmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            pos_encoding=pos_type
        )        
        embeddings = embedding_layer(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {embeddings.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")        
        analyzer = EmbeddingAnalyzer(embedding_layer)
        stats = analyzer.analyze_embedding_distribution()
        print(f"Embedding statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")    
    print(f"\nTesting Adaptive Embeddings:")
    adaptive_emb = AdaptiveEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        cutoffs=[200, 500],
        div_val=2.0
    )    
    embeddings = adaptive_emb(input_ids)
    print(f"Adaptive embedding output shape: {embeddings.shape}")    
    print(f"\nTesting masking functions:")
    input_ids_padded = input_ids.clone()
    input_ids_padded[:, -10:] = 0
    padding_mask = create_padding_mask(input_ids_padded, padding_idx=0)
    causal_mask = create_causal_mask(seq_len, input_ids.device)
    attention_mask = create_attention_mask(input_ids_padded, padding_idx=0, causal=True)    
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Padding positions: {padding_mask.sum().item()}")
    print(f"Causal mask positions: {causal_mask.sum().item()}")
    print(f"Combined mask positions: {attention_mask.sum().item()}")    
    print("\nAll embedding tests completed successfully!")
if __name__ == "__main__":
    test_embeddings()
