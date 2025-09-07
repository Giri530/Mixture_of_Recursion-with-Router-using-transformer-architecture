## MixtureofRecursionwithRouter
A transformer-based small-scale language model optimized for technical content, featuring a custom tokenizer and a recursive transformer architecture with an adaptive router for dynamic computation steps. Designed for efficient training (4-5 hours) and inference on technical datasets, this model excels in processing code snippets, mathematical expressions, and technical conversations.

## Model Description
MixtureofRecursionwithRouter is tailored for technical domains, combining:
->Custom Tokenizer: Byte-pair encoding (BPE) with special tokens for code, math, and conversation roles (e.g., <user>, <assistant>).
->Adaptive Embeddings: Token embeddings with configurable positional encodings (learned, sinusoidal, or RoPE).
->Recursive Transformer: Multi-layered architecture with a RecursionRouter to dynamically adjust computation steps based on input complexity.
->Ultra-Fast Training: Optimized for low loss (<2.0) and perplexity (<12) using mixed precision and cosine scheduling.

## Model Details

->Vocabulary Size: 32,000 
->Embedding Dimension: 384 
->Number of Layers: 6 
->Attention Heads: 6 
->Max Sequence Length: 128 
->Positional Encoding: Learned (default, supports sinusoidal or RoPE)
->Training Objective: Causal language modeling with cross-entropy loss

## Performance:
->Validation Loss: 2.07
->Validation Perplexity: 7.9


## Optimizer: AdamW with cosine learning rate scheduling
## Hardware: Trained on GPU (CUDA-compatible) or CPU
## Training Time: ~4-5 hours on a single GPU
## Parameters: 10M (exact count via count_parameters(model))

## Installation
Requires Python 3.8+ and the following dependencies:
->pip install torch numpy tqdm

## Clone the repository:
git clone https://huggingface.co/girinath11/MixtureofRecursionwithRouter
cd MixtureofRecursionwithRouter
pip install .

## Usage
## Loading the Model
from model_slm import MixtureOfRecursions
from custom_tokenizer import TechnicalTokenizer
import torch

# Load tokenizer
tokenizer = TechnicalTokenizer()
tokenizer.load("path/to/tokenizer")

# Initialize model
model = MixtureOfRecursions(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=384,
    n_layers=6,
    n_heads=6,
    max_seq_len=128,
    padding_idx=tokenizer.vocab.get('<pad>', 0)
)

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

Text Generation
from model_slm import TextGenerator

# Initialize generator
generator = TextGenerator(model, tokenizer, max_length=128, device=device)

# Generate text
prompt = "Write a Python function to compute the Fibonacci sequence."
response = generator.generate(
    prompt,
    method="nucleus",
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=100
)
print(response)

## Training
Prepare a dataset in .txt format and run:
python train.py \
    --train_file path/to/train.txt \
    --val_file path/to/val.txt \
    --tokenizer_dir path/to/tokenizer \
    --max_examples 50000 \
    --d_model 384 \
    --n_layers 6 \
    --n_heads 6 \
    --max_seq_len 128 \
    --epochs 15 \
    --batch_size 16

The training script uses mixed precision, gradient accumulation, and a cosine learning rate scheduler to achieve a validation loss of 2.07 and perplexity of 7.9 in 4-5 hours.
## Dataset
The model is trained on technical conversation datasets (.txt). The FastTechnicalTextDataset class applies filters:
->Text length: 50–400 characters
->Minimum 8 words
->No URLs or excessive punctuation
->Deduplication via hashing
->Maximum 50,000 examples

## Example JSONL Format:
{"messages": [{"role": "user", "content": "How does backpropagation work?"}, {"role": "assistant", "content": "Backpropagation is..."}]}

## Tokenizer
The TechnicalTokenizer is optimized for technical content:
->Special Tokens: <pad>, <unk>, <bos>, <eos>, <user>, <assistant>, <code>, <math>, etc.
->BPE: Subword tokenization with a vocabulary of 32,000.
->Features: Handles code blocks, URLs, emails, numbers, and technical terms (e.g., "algorithm", "neural").
N->ormalization: Unicode NFKC normalization.

## To train the tokenizer:
from custom_tokenizer import train_tokenizer_from_files

train_tokenizer_from_files(
    file_paths=["path/to/train.txt"],
    vocab_size=32000,
    min_freq=2,
    output_dir="tokenizer"
)

## Model Architecture
The MixtureofRecursionwithRouter model is a transformer-based architecture specifically designed for technical content, incorporating several innovative components to enhance performance and efficiency:

## Embedding Layer (TechEmbeddingLayer):

Combines token embeddings with configurable positional encodings (learned by default, with support for sinusoidal or RoPE).
Uses a d_model of 384 for compact yet expressive representations.
Applies layer normalization and dropout (0.1) for regularization.
Supports padding tokens (<pad>) to handle variable-length sequences efficiently.


## Attention Mechanism (MultiHeadAttention):

Implements multi-head self-attention with 6 heads, each handling a subspace of the 384-dimensional input.
Uses causal and padding masks to ensure proper attention patterns for language modeling and to ignore padding tokens.
Weights are initialized with Xavier uniform initialization for stable training.
Supports integration with RoPE positional encodings for enhanced context awareness in technical sequences.


## Recursive Transformer Layers (RecursiveTransformerLayer):

Consists of 6 layers, each incorporating a MultiHeadAttention module, a FeedForward network, and two layer normalization steps.
RecursionRouter that dynamically determines the number of recursive computation steps (up to 4) based on input complexity.
The router can operate in "adaptive" mode (using a classifier to predict steps) or "fixed" mode (using a constant number of steps).
Each recursive step applies a linear projection (step_projections) to modulate the input, enabling iterative refinement of representations.
Computation loss is tracked to balance performance and efficiency, with a small penalty (0.0001) applied to encourage efficient routing.


## Feedforward Network (FeedForward):

Position-wise feedforward network with GELU activation and a hidden dimension of 2048.
Applies dropout (0.1) to prevent overfitting and Xavier initialization for stable training.
Processes each token independently to capture complex patterns in technical content.


## Output Layer:

A linear layer maps the 384-dimensional hidden states to the vocabulary size (32,000).
Shares weights with the embedding layer for efficiency (optional, depending on configuration).
Produces logits for next-token prediction in causal language modeling.


## Adaptive Routing (RecursionRouter):

A unique feature that evaluates input complexity using a small neural network (linear layer, GELU, dropout, and softmax).
Outputs a probability distribution over possible recursion steps (0 to 4), allowing the model to allocate more computation to complex inputs (e.g., code or math) and fewer to simpler ones.
Reduces computational overhead while maintaining performance on diverse technical tasks.

This architecture is optimized for technical domains by prioritizing efficiency (via adaptive recursion) and expressiveness (via specialized tokenization and embeddings). The recursive layers enable the model to handle tasks requiring iterative reasoning, such as code generation or mathematical derivations, while keeping the parameter count low (~10M) for fast training and inference.
## Evaluation
Evaluated on a validation set with:

Loss: 2.07
Perplexity: 7.9

Validation is performed every 500 steps (configurable). Example metrics:
{
  "epoch": 15,
  "train_loss": 1.85,
  "train_ppl": 6.35,
  "val_loss": 2.07,
  "val_ppl": 7.9,
  "epoch_time_min": 12.5
}

## Checkpoints
Checkpoints are saved in the checkpoints directory when a new best validation loss is achieved. Each checkpoint includes:

Model state
Optimizer state
Scaler state
Metrics

## To load a checkpoint:
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

## Limitations
->Sequence Length: Limited to 128 tokens (configurable, but longer sequences increase memory usage).
->Dataset Size: Optimized for 50,000 examples to ensure fast training.
->Domain: Tailored for technical content; may not generalize to non-technical text.
->Hardware: Best performance on GPU; CPU training is slower.

## License
This model is licensed under the Apache-2.0 License. See the LICENSE file for details.

## Acknowledgments
->Built using PyTorch.
->Inspired by transformer architectures and BPE tokenization.
->Optimized for technical content with insights from domain-specific language models.
