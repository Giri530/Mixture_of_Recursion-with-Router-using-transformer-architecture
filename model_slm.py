import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from embeddings import TechEmbeddingLayer, create_padding_mask, create_causal_mask
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism optimized for technical content"""    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)    
    def forward(self, query, key, value, mask=None, pos_encoding=None):
        batch_size, seq_len, d_model = query.size()        
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)        
        if pos_encoding is not None:
            Q, K = pos_encoding(Q, K)        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, seq_len, seq_len)
            scores.masked_fill_(mask, float('-inf'))        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(attended)        
        return output
class FeedForward(nn.Module):
    """Position-wise feed forward network with GELU activation"""    
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)        
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)    
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class RecursionRouter(nn.Module):
    """Router to decide recursion steps for different types of technical problems"""    
    def __init__(self, d_model, max_steps=4, router_type="adaptive"):
        super(RecursionRouter, self).__init__()
        self.max_steps = max_steps
        self.router_type = router_type        
        if router_type == "adaptive":
            self.complexity_classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, max_steps + 1),
                nn.Softmax(dim=-1)
            )
        elif router_type == "fixed":
            self.fixed_steps = max_steps    
    def forward(self, x):
        if self.router_type == "adaptive":
            seq_repr = x.mean(dim=1)
            step_probs = self.complexity_classifier(seq_repr)
            steps = torch.argmax(step_probs, dim=-1)
            return steps
        return self.fixed_steps
class RecursiveTransformerLayer(nn.Module):
    """Transformer layer with recursive computation capability"""    
    def __init__(self, d_model, n_heads, dim_feedforward, max_steps=4, 
                 dropout=0.1, router_type="adaptive"):
        super(RecursiveTransformerLayer, self).__init__()        
        self.max_steps = max_steps
        self.d_model = d_model        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feedforward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.router = RecursionRouter(d_model, max_steps, router_type)
        self.step_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(max_steps)
        ])    
    def forward(self, x, mask=None, pos_encoding=None):
        steps = self.router(x)
        if isinstance(steps, int):
            num_steps = min(steps, self.max_steps)
            return self._recursive_forward_fixed(x, mask, num_steps, pos_encoding)
        return self._recursive_forward_adaptive(x, mask, steps, pos_encoding)    
    def _recursive_forward_fixed(self, x, mask, num_steps, pos_encoding):
        device = x.device
        batch_size = x.shape[0]
        computation_loss = torch.tensor(0.0, device=device)        
        for step in range(num_steps):
            step_input = self.step_projections[step](x) if step < len(self.step_projections) else x
            attended = self.attention(step_input, step_input, step_input, mask, pos_encoding)
            x = self.norm1(x + self.dropout(attended))
            fed_forward = self.feedforward(x)
            x = self.norm2(x + self.dropout(fed_forward))
            computation_loss += torch.tensor(0.1, device=device) * batch_size        
        return x, computation_loss    
    def _recursive_forward_adaptive(self, x, mask, steps, pos_encoding):
        batch_size, seq_len, d_model = x.shape
        device = x.device
        max_batch_steps = int(steps.max().item())
        computation_loss = torch.tensor(0.0, device=device)        
        active_batches = torch.ones(batch_size, device=device, dtype=torch.bool)
        for step in range(max_batch_steps):
            step_mask = (steps > step) & active_batches
            if not step_mask.any():
                break                
            step_input = self.step_projections[step](x) if step < len(self.step_projections) else x
            attended = self.attention(step_input, step_input, step_input, mask, pos_encoding)
            attended = torch.where(step_mask.unsqueeze(-1).unsqueeze(-1), attended, torch.zeros_like(attended))
            x = self.norm1(x + self.dropout(attended))
            fed_forward = self.feedforward(x)
            fed_forward = torch.where(step_mask.unsqueeze(-1).unsqueeze(-1), fed_forward, torch.zeros_like(fed_forward))
            x = self.norm2(x + self.dropout(fed_forward))
            computation_loss += torch.tensor(0.1, device=device) * step_mask.sum()
            active_batches &= (steps > step)        
        return x, computation_loss
class MixtureOfRecursions(nn.Module):
    """Main model with mixture of recursive transformer layers"""    
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, 
                 max_steps=4, dim_feedforward=2048, dropout=0.1, 
                 max_seq_len=512, router_type="adaptive", padding_idx=0):
        super(MixtureOfRecursions, self).__init__()        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx        
        self.embeddings = TechEmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
            pos_encoding="learned"
        )        
        self.layers = nn.ModuleList([
            RecursiveTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                max_steps=max_steps,
                dropout=dropout,
                router_type=router_type
            ) for _ in range(n_layers)
        ])        
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape        
        padding_mask = create_padding_mask(input_ids, self.padding_idx) if attention_mask is None else (attention_mask == 0)
        causal_mask = create_causal_mask(seq_len, input_ids.device)
        padding_mask = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        combined_mask = padding_mask | causal_mask.unsqueeze(0)        
        x = self.embeddings(input_ids)
        pos_encoding = self.embeddings.get_positional_encoding()        
        device = x.device
        total_computation_loss = torch.tensor(0.0, device=device)
        for layer in self.layers:
            x, comp_loss = layer(x, combined_mask, pos_encoding)
            total_computation_loss += comp_loss        
        x = self.final_norm(x)
        logits = self.lm_head(x)        
        return logits, total_computation_loss    
    def generate_step(self, input_ids, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            last_logits = logits[:, -1, :] / temperature            
            if top_k is not None:
                indices_to_remove = last_logits < torch.topk(last_logits, top_k)[0][..., -1, None]
                last_logits[indices_to_remove] = float('-inf')            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                last_logits[indices_to_remove] = float('-inf')            
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token
class TextGenerator:
    """Text generation utility for the tech model"""    
    def __init__(self, model, tokenizer, max_length=100, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device if device else next(model.parameters()).device
        self.model.to(self.device)
        self.eos_token_id = tokenizer.vocab.get('<|endoftext|>', -1)
        self.assistant_token_id = tokenizer.vocab.get('<|assistant|>', -1)    
    def generate(self, prompt, method="nucleus", temperature=1.0, top_k=50, top_p=0.9, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_length        
        input_text = f"<|user|> {prompt}"
        input_ids = self.tokenizer.encode_ids(input_text, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)        
        self.model.eval()
        generated_ids = []        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if input_tensor.size(1) > self.max_length:
                    input_tensor = input_tensor[:, -self.max_length:]                
                # Generate next token
                if method == "greedy":
                    next_token = self._greedy_generate(input_tensor)
                elif method == "sample":
                    next_token = self._sample_generate(input_tensor, temperature)
                elif method == "top_k":
                    next_token = self._top_k_generate(input_tensor, temperature, top_k)
                elif method == "nucleus" or method == "top_p":
                    next_token = self._nucleus_generate(input_tensor, temperature, top_p)
                else:
                    raise ValueError(f"Unknown generation method: {method}")                
                next_token_id = next_token.item()
                generated_ids.append(next_token_id)
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                if next_token_id == self.eos_token_id or (self.assistant_token_id != -1 and next_token_id == self.assistant_token_id):
                    break        
        # Decode the full sequence
        full_ids = input_ids + generated_ids
        full_text = self.tokenizer.decode_ids(full_ids, skip_special_tokens=False)        
        # Extract assistant response
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1].split("<|endoftext|>")[0].strip()
        else:
            response = full_text.split("<|endoftext|>")[0].strip()        
        return response if response else "No response generated."    
    def _greedy_generate(self, input_tensor):
        logits, _ = self.model(input_tensor)
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)    
    def _sample_generate(self, input_tensor, temperature):
        logits, _ = self.model(input_tensor)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)   
    def _top_k_generate(self, input_tensor, temperature, top_k):
        logits, _ = self.model(input_tensor)
        logits = logits[:, -1, :] / temperature
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)
        return top_k_indices.gather(-1, next_token_idx)    
    def _nucleus_generate(self, input_tensor, temperature, top_p):
        return self.model.generate_step(input_tensor, temperature, top_p=top_p)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
def main():
    vocab_size = 10000
    d_model = 512
    n_layers = 6
    n_heads = 8
    seq_len = 128
    batch_size = 4    
    print("Initializing MixtureOfRecursions model...")
    model = MixtureOfRecursions(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_steps=4,
        dim_feedforward=2048,
        dropout=0.1,
        router_type="adaptive"
    )    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")    
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -10:] = 0    
    print(f"Input shape: {input_ids.shape}")
    logits, comp_loss = model(input_ids, attention_mask)    
    print(f"Output logits shape: {logits.shape}")
    print(f"Computation loss: {comp_loss}")
    print(f"Expected logits shape: ({batch_size}, {seq_len}, {vocab_size})")    
    print("\nTesting generation step...")
    next_token = model.generate_step(input_ids[:1], temperature=0.8, top_p=0.9)
    print(f"Generated next token: {next_token}")    
    print("\nModel test completed successfully!")
if __name__ == "__main__":
    main()
