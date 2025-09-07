import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import json
import argparse
import time
import math
import glob
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import gc
from collections import defaultdict
import multiprocessing
# Import custom modules
try:
    from model_slm import MixtureOfRecursions, count_parameters, TextGenerator
    from custom_tokenizer import TechnicalTokenizer
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)
class FastTechnicalTextDataset(Dataset):
    """Ultra-fast dataset with aggressive optimizations for 4-5hr training"""    
    def __init__(self, data_file: str, tokenizer: TechnicalTokenizer, max_length: int = 128, max_examples: int = 50000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.vocab.get('<pad>', 0)
        self.max_examples = max_examples        
        print(f"FAST DATASET LOADING")
        print(f"Data file: {data_file}")
        print(f"Max sequence length: {max_length}")
        print(f"Max examples: {max_examples}")        
        start_time = time.time()
        self.examples = []
        self._fast_load_data(data_file)        
        load_time = time.time() - start_time
        print(f" Loaded {len(self.examples)} examples in {load_time:.1f}s")        
        self._tensorize_data()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None    
    def _fast_load_data(self, data_file: str):
        print("🔍 Fast reading file...")        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()        
        print(f"File has {len(lines)} lines")        
        good_examples = []
        seen_hashes = set()        
        for line in lines[:self.max_examples * 3]:
            line = line.strip()
            if (50 <= len(line) <= 400 and
                line.count(' ') >= 8 and
                not line.lower().startswith(('http', 'www', 'ftp')) and
                line.count('.') <= len(line) * 0.1):                
                line_hash = hash(line[:100])
                if line_hash not in seen_hashes:
                    seen_hashes.add(line_hash)
                    good_examples.append(line)
                    if len(good_examples) >= self.max_examples:
                        break        
        print(f"After fast filtering: {len(good_examples)} quality examples")        
        batch_size = 1000
        for i in range(0, len(good_examples), batch_size):
            batch = good_examples[i:i+batch_size]
            for line in batch:
                try:
                    if not line.endswith('<|endoftext|>'):
                        line += ' <|endoftext|>'                    
                    tokens = self.tokenizer.encode_ids(line, add_special_tokens=True)
                    if 30 <= len(tokens) <= self.max_length:
                        if len(tokens) < self.max_length:
                            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
                        self.examples.append(tokens)
                except:
                    continue
            if i % 5000 == 0:
                print(f"Processed {len(self.examples)} examples...")        
        print(f"Final dataset: {len(self.examples)} examples")    
    def _tensorize_data(self):
        print("Pre-tensorizing data for maximum speed...")
        seq_len = self.max_length - 1        
        tensorized_examples = []
        for tokens in self.examples:
            if len(tokens) < self.max_length:
                continue            
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            targets = torch.tensor(tokens[1:], dtype=torch.long)            
            original_len = next((i for i, x in enumerate(tokens) if x == self.pad_token_id), self.max_length)
            mask_len = min(original_len, seq_len)
            attention_mask = torch.zeros(seq_len, dtype=torch.long)
            attention_mask[:mask_len] = 1            
            tensorized_examples.append({
                'input_ids': input_ids,
                'targets': targets,
                'attention_mask': attention_mask
            })
        self.examples = tensorized_examples
        print("All data pre-tensorized")    
    def __len__(self):
        return len(self.examples)    
    def __getitem__(self, idx):
        return self.examples[idx]
class FastCosineScheduler:
    def __init__(self, optimizer, total_steps: int, warmup_ratio: float = 0.05):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
class UltraFastTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset=None, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)        
        self._fast_init_weights()
        self._setup_fast_optimizer()        
        epochs = self.config.get('epochs', 15)
        batch_size = self.config.get('batch_size', 16)
        total_steps = len(train_dataset) // batch_size * epochs
        self.scheduler = FastCosineScheduler(self.optimizer, total_steps)
        self.scaler = GradScaler()        
        self.global_step = 0
        self.best_loss = float('inf')
        self.grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
        self.eval_every = self.config.get('eval_every', 500)    
    def _fast_init_weights(self):
        def fast_init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        self.model.apply(fast_init)    
    def _setup_fast_optimizer(self):
        lr = self.config.get('learning_rate', 5e-4)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.01, eps=1e-6)    
    def compute_fast_loss(self, logits, targets, mask):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1).bool()
        if not mask_flat.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
        return loss    
    def train_epoch_fast(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, miniters=50)
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            mask = batch['attention_mask'].to(self.device, non_blocking=True)            
            with autocast():
                logits, comp_loss = self.model(input_ids, mask)
                lm_loss = self.compute_fast_loss(logits, targets, mask)
                total_loss_step = lm_loss + 0.0001 * comp_loss
                if self.grad_accum_steps > 1:
                    total_loss_step = total_loss_step / self.grad_accum_steps            
            self.scaler.scale(total_loss_step).backward()
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1            
            total_loss += lm_loss.item()
            num_batches += 1
            if batch_idx % 100 == 0:
                current_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': f"{current_loss:.3f}", 'ppl': f"{math.exp(min(current_loss, 10)):.1f}"})
            if batch_idx % 200 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()       
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'perplexity': math.exp(min(avg_loss, 10)), 'epoch_time_min': epoch_time / 60}    
    def validate_fast(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        max_val_batches = min(100, len(dataloader))        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_val_batches:
                    break
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                mask = batch['attention_mask'].to(self.device, non_blocking=True)
                with autocast():
                    logits, _ = self.model(input_ids, mask)
                    loss = self.compute_fast_loss(logits, targets, mask)
                total_loss += loss.item()
                num_batches += 1        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'perplexity': math.exp(min(avg_loss, 10))}    
    def save_checkpoint_fast(self, epoch: int, metrics: Dict, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'scaler_state_dict': self.scaler.state_dict()
            }
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best! Loss: {val_loss:.4f}")
            return best_path
        return None    
    def train_ultra_fast(self, num_epochs: int = 15, batch_size: int = 16):
        print(f"\n ULTRA-FAST TRAINING")
        print(f" Target: Loss < 2.0, PPL < 12")
        print(f" Time target: 4-5 hours")
        print(f" Epochs: {num_epochs}")
        print(f" Batch size: {batch_size}")
        print("-" * 60)        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )        
        total_start_time = time.time()
        history = []        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            print(f"\n EPOCH {epoch}/{num_epochs}")
            train_metrics = self.train_epoch_fast(epoch, train_loader)            
            val_metrics = {}
            if val_loader and (epoch % 2 == 0 or epoch == num_epochs):
                val_metrics = self.validate_fast(val_loader)            
            epoch_time = time.time() - epoch_start
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_ppl': train_metrics['perplexity'],
                'epoch_time_min': epoch_time / 60
            }
            if val_metrics:
                epoch_info.update({'val_loss': val_metrics['loss'], 'val_ppl': val_metrics['perplexity']})
            history.append(epoch_info)            
            elapsed_hours = (time.time() - total_start_time) / 3600
            remaining_hours = elapsed_hours * (num_epochs - epoch) / epoch            
            print(f"\n EPOCH {epoch} RESULTS:")
            print(f" Epoch time: {epoch_time/60:.1f} min")
            print(f" Total elapsed: {elapsed_hours:.1f}h")
            print(f" Est. remaining: {remaining_hours:.1f}h")
            print(f" Train Loss: {train_metrics['loss']:.4f}")
            print(f" Train PPL: {train_metrics['perplexity']:.1f}")
            if val_metrics:
                print(f" Val Loss: {val_metrics['loss']:.4f}")
                print(f" Val PPL: {val_metrics['perplexity']:.1f}")            
            current_loss = val_metrics.get('loss', train_metrics['loss'])
            current_ppl = val_metrics.get('perplexity', train_metrics['perplexity'])
            if current_loss < 2.0 and current_ppl < 12:
                print(f" TARGETS ACHIEVED!")
                print(f" Loss: {current_loss:.4f} < 2.0")
                print(f" PPL: {current_ppl:.1f} < 12")            
            combined_metrics = {**train_metrics}
            if val_metrics:
                combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.save_checkpoint_fast(epoch, combined_metrics)           
            torch.cuda.empty_cache()
            gc.collect()            
            if current_loss < 1.8 and current_ppl < 10:
                print(f"EARLY STOPPING - Excellent performance achieved!")
                break        
        total_time = time.time() - total_start_time
        print(f"\n TRAINING COMPLETED!")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f" Best loss: {self.best_loss:.4f}")
        return history
def run_ultra_fast_training():
    parser = argparse.ArgumentParser(description="Ultra-Fast Training for 4-5 Hours")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--max_examples", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=500)    
    args = parser.parse_args()    
    torch.manual_seed(42)
    np.random.seed(42)    
    print("Training My Model")
    print("-" * 50)    
    if args.train_file is None:
        patterns = ["*train*.txt", "*_train.txt"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
            files.extend(glob.glob(f"split_data/{pattern}"))
            files.extend(glob.glob(f"data/{pattern}"))
        if files:
            args.train_file = files[0]
            print(f"Found: {args.train_file}")
        else:
            print(" No training files found!")
            return 1    
    tokenizer = TechnicalTokenizer()
    try:
        tokenizer.load(args.tokenizer_dir)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f" Tokenizer error: {e}")
        return 1    
    print(" Creating ultra-fast dataset...")
    train_dataset = FastTechnicalTextDataset(
        args.train_file, tokenizer, args.max_seq_len, args.max_examples
    )    
    val_dataset = None
    if args.val_file and os.path.exists(args.val_file):
        val_dataset = FastTechnicalTextDataset(
            args.val_file, tokenizer, args.max_seq_len, max_examples=5000
        )    
    model = MixtureOfRecursions(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len - 1, # Pass the actual sequence length to the model
        padding_idx=tokenizer.vocab.get('<pad>', 0)
    )    
    config = {
        'learning_rate': args.learning_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'eval_every': args.eval_every,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }    
    trainer = UltraFastTrainer(model, tokenizer, train_dataset, val_dataset, config)
    print(f"\n START TRAINING")
    results = trainer.train_ultra_fast(args.epochs, args.batch_size)    
    with open('ultra_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)    
    print("\n Training Completed!")
    print(" Results saved to: ultra_fast_results.json")
    return 0
if __name__ == "__main__":
    exit(run_ultra_fast_training())import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import json
import argparse
import time
import math
import glob
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import gc
from collections import defaultdict
import multiprocessing
# Import custom modules
try:
    from model_slm import MixtureOfRecursions, count_parameters, TextGenerator
    from custom_tokenizer import TechnicalTokenizer
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)
class FastTechnicalTextDataset(Dataset):
    """Ultra-fast dataset with aggressive optimizations for 4-5hr training"""    
    def __init__(self, data_file: str, tokenizer: TechnicalTokenizer, max_length: int = 128, max_examples: int = 50000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.vocab.get('<pad>', 0)
        self.max_examples = max_examples        
        print(f"FAST DATASET LOADING")
        print(f"Data file: {data_file}")
        print(f"Max sequence length: {max_length}")
        print(f"Max examples: {max_examples}")        
        start_time = time.time()
        self.examples = []
        self._fast_load_data(data_file)        
        load_time = time.time() - start_time
        print(f" Loaded {len(self.examples)} examples in {load_time:.1f}s")        
        self._tensorize_data()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None    
    def _fast_load_data(self, data_file: str):
        print("🔍 Fast reading file...")        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()        
        print(f"File has {len(lines)} lines")        
        good_examples = []
        seen_hashes = set()        
        for line in lines[:self.max_examples * 3]:
            line = line.strip()
            if (50 <= len(line) <= 400 and
                line.count(' ') >= 8 and
                not line.lower().startswith(('http', 'www', 'ftp')) and
                line.count('.') <= len(line) * 0.1):                
                line_hash = hash(line[:100])
                if line_hash not in seen_hashes:
                    seen_hashes.add(line_hash)
                    good_examples.append(line)
                    if len(good_examples) >= self.max_examples:
                        break        
        print(f"After fast filtering: {len(good_examples)} quality examples")        
        batch_size = 1000
        for i in range(0, len(good_examples), batch_size):
            batch = good_examples[i:i+batch_size]
            for line in batch:
                try:
                    if not line.endswith('<|endoftext|>'):
                        line += ' <|endoftext|>'                    
                    tokens = self.tokenizer.encode_ids(line, add_special_tokens=True)
                    if 30 <= len(tokens) <= self.max_length:
                        if len(tokens) < self.max_length:
                            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
                        self.examples.append(tokens)
                except:
                    continue
            if i % 5000 == 0:
                print(f"Processed {len(self.examples)} examples...")        
        print(f"Final dataset: {len(self.examples)} examples")    
    def _tensorize_data(self):
        print("Pre-tensorizing data for maximum speed...")
        seq_len = self.max_length - 1        
        tensorized_examples = []
        for tokens in self.examples:
            if len(tokens) < self.max_length:
                continue            
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            targets = torch.tensor(tokens[1:], dtype=torch.long)            
            original_len = next((i for i, x in enumerate(tokens) if x == self.pad_token_id), self.max_length)
            mask_len = min(original_len, seq_len)
            attention_mask = torch.zeros(seq_len, dtype=torch.long)
            attention_mask[:mask_len] = 1            
            tensorized_examples.append({
                'input_ids': input_ids,
                'targets': targets,
                'attention_mask': attention_mask
            })
        self.examples = tensorized_examples
        print("All data pre-tensorized")    
    def __len__(self):
        return len(self.examples)    
    def __getitem__(self, idx):
        return self.examples[idx]
class FastCosineScheduler:
    def __init__(self, optimizer, total_steps: int, warmup_ratio: float = 0.05):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
class UltraFastTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset=None, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)        
        self._fast_init_weights()
        self._setup_fast_optimizer()        
        epochs = self.config.get('epochs', 15)
        batch_size = self.config.get('batch_size', 16)
        total_steps = len(train_dataset) // batch_size * epochs
        self.scheduler = FastCosineScheduler(self.optimizer, total_steps)
        self.scaler = GradScaler()        
        self.global_step = 0
        self.best_loss = float('inf')
        self.grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
        self.eval_every = self.config.get('eval_every', 500)    
    def _fast_init_weights(self):
        def fast_init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        self.model.apply(fast_init)    
    def _setup_fast_optimizer(self):
        lr = self.config.get('learning_rate', 5e-4)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.01, eps=1e-6)    
    def compute_fast_loss(self, logits, targets, mask):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1).bool()
        if not mask_flat.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
        return loss    
    def train_epoch_fast(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, miniters=50)
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            mask = batch['attention_mask'].to(self.device, non_blocking=True)            
            with autocast():
                logits, comp_loss = self.model(input_ids, mask)
                lm_loss = self.compute_fast_loss(logits, targets, mask)
                total_loss_step = lm_loss + 0.0001 * comp_loss
                if self.grad_accum_steps > 1:
                    total_loss_step = total_loss_step / self.grad_accum_steps            
            self.scaler.scale(total_loss_step).backward()
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1            
            total_loss += lm_loss.item()
            num_batches += 1
            if batch_idx % 100 == 0:
                current_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': f"{current_loss:.3f}", 'ppl': f"{math.exp(min(current_loss, 10)):.1f}"})
            if batch_idx % 200 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()       
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'perplexity': math.exp(min(avg_loss, 10)), 'epoch_time_min': epoch_time / 60}    
    def validate_fast(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        max_val_batches = min(100, len(dataloader))        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_val_batches:
                    break
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                mask = batch['attention_mask'].to(self.device, non_blocking=True)
                with autocast():
                    logits, _ = self.model(input_ids, mask)
                    loss = self.compute_fast_loss(logits, targets, mask)
                total_loss += loss.item()
                num_batches += 1        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'perplexity': math.exp(min(avg_loss, 10))}    
    def save_checkpoint_fast(self, epoch: int, metrics: Dict, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'scaler_state_dict': self.scaler.state_dict()
            }
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best! Loss: {val_loss:.4f}")
            return best_path
        return None    
    def train_ultra_fast(self, num_epochs: int = 15, batch_size: int = 16):
        print(f"\n ULTRA-FAST TRAINING")
        print(f" Target: Loss < 2.0, PPL < 12")
        print(f" Time target: 4-5 hours")
        print(f" Epochs: {num_epochs}")
        print(f" Batch size: {batch_size}")
        print("-" * 60)        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )        
        total_start_time = time.time()
        history = []        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            print(f"\n EPOCH {epoch}/{num_epochs}")
            train_metrics = self.train_epoch_fast(epoch, train_loader)            
            val_metrics = {}
            if val_loader and (epoch % 2 == 0 or epoch == num_epochs):
                val_metrics = self.validate_fast(val_loader)            
            epoch_time = time.time() - epoch_start
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_ppl': train_metrics['perplexity'],
                'epoch_time_min': epoch_time / 60
            }
            if val_metrics:
                epoch_info.update({'val_loss': val_metrics['loss'], 'val_ppl': val_metrics['perplexity']})
            history.append(epoch_info)            
            elapsed_hours = (time.time() - total_start_time) / 3600
            remaining_hours = elapsed_hours * (num_epochs - epoch) / epoch            
            print(f"\n EPOCH {epoch} RESULTS:")
            print(f" Epoch time: {epoch_time/60:.1f} min")
            print(f" Total elapsed: {elapsed_hours:.1f}h")
            print(f" Est. remaining: {remaining_hours:.1f}h")
            print(f" Train Loss: {train_metrics['loss']:.4f}")
            print(f" Train PPL: {train_metrics['perplexity']:.1f}")
            if val_metrics:
                print(f" Val Loss: {val_metrics['loss']:.4f}")
                print(f" Val PPL: {val_metrics['perplexity']:.1f}")            
            current_loss = val_metrics.get('loss', train_metrics['loss'])
            current_ppl = val_metrics.get('perplexity', train_metrics['perplexity'])
            if current_loss < 2.0 and current_ppl < 12:
                print(f" TARGETS ACHIEVED!")
                print(f" Loss: {current_loss:.4f} < 2.0")
                print(f" PPL: {current_ppl:.1f} < 12")            
            combined_metrics = {**train_metrics}
            if val_metrics:
                combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.save_checkpoint_fast(epoch, combined_metrics)           
            torch.cuda.empty_cache()
            gc.collect()            
            if current_loss < 1.8 and current_ppl < 10:
                print(f"EARLY STOPPING - Excellent performance achieved!")
                break        
        total_time = time.time() - total_start_time
        print(f"\n TRAINING COMPLETED!")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f" Best loss: {self.best_loss:.4f}")
        return history
def run_ultra_fast_training():
    parser = argparse.ArgumentParser(description="Ultra-Fast Training for 4-5 Hours")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--max_examples", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=500)    
    args = parser.parse_args()    
    torch.manual_seed(42)
    np.random.seed(42)    
    print("Training My Model")
    print("-" * 50)    
    if args.train_file is None:
        patterns = ["*train*.txt", "*_train.txt"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
            files.extend(glob.glob(f"split_data/{pattern}"))
            files.extend(glob.glob(f"data/{pattern}"))
        if files:
            args.train_file = files[0]
            print(f"Found: {args.train_file}")
        else:
            print(" No training files found!")
            return 1    
    tokenizer = TechnicalTokenizer()
    try:
        tokenizer.load(args.tokenizer_dir)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f" Tokenizer error: {e}")
        return 1    
    print(" Creating ultra-fast dataset...")
    train_dataset = FastTechnicalTextDataset(
        args.train_file, tokenizer, args.max_seq_len, args.max_examples
    )    
    val_dataset = None
    if args.val_file and os.path.exists(args.val_file):
        val_dataset = FastTechnicalTextDataset(
            args.val_file, tokenizer, args.max_seq_len, max_examples=5000
        )    
    model = MixtureOfRecursions(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len - 1, # Pass the actual sequence length to the model
        padding_idx=tokenizer.vocab.get('<pad>', 0)
    )    
    config = {
        'learning_rate': args.learning_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'eval_every': args.eval_every,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }    
    trainer = UltraFastTrainer(model, tokenizer, train_dataset, val_dataset, config)
    print(f"\n START TRAINING")
    results = trainer.train_ultra_fast(args.epochs, args.batch_size)    
    with open('ultra_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)    
    print("\n Training Completed!")
    print(" Results saved to: ultra_fast_results.json")
    return 0
if __name__ == "__main__":
    exit(run_ultra_fast_training())
