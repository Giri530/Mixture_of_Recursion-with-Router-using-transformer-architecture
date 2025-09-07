import os
import json
import pickle
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Set, Optional, Tuple
import re
import unicodedata
class TechnicalTokenizer:
    """
    Custom tokenizer optimized for technical content and conversations
    """    
    def __init__(self, vocab_size: int = 32000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq       
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<system>': 4,
            '<user>': 5,
            '<assistant>': 6,
            '<|endoftext|>': 7,
            '<|newline|>': 8,
            '<|tab|>': 9,
            '<|code|>': 10,
            '<|/code|>': 11,
            '<|math|>': 12,
            '<|/math|>': 13
        }       
        self.vocab = {}
        self.id_to_token = {}
        self.token_frequencies = Counter()
        self.bpe_merges = []
        self.bpe_cache = {}       
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')        
        self.technical_terms = {
            'function', 'variable', 'array', 'object', 'class', 'method', 'parameter',
            'return', 'import', 'export', 'async', 'await', 'promise', 'callback',
            'algorithm', 'datatype', 'boolean', 'integer', 'string', 'float',
            'javascript', 'python', 'java', 'cpp', 'html', 'css', 'sql',
            'api', 'json', 'xml', 'http', 'https', 'rest', 'graphql',
            'equation', 'formula', 'theorem', 'proof', 'hypothesis',
            'derivative', 'integral', 'matrix', 'vector', 'polynomial',
            'probability', 'statistics', 'correlation', 'regression',
            'neural', 'network', 'model', 'training', 'validation', 'test',
            'accuracy', 'precision', 'recall', 'f1score', 'loss', 'gradient',
            'backpropagation', 'forward', 'layer', 'neuron', 'weight', 'bias',
            'transformer', 'attention', 'embedding', 'tokenization',
            'database', 'server', 'client', 'protocol', 'encryption', 'security',
            'authentication', 'authorization', 'deployment', 'docker', 'kubernetes',
            'microservice', 'architecture', 'scalability', 'performance'
        }       
        self._init_vocab()   
    def _init_vocab(self):
        self.vocab = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}    
    def normalize_text(self, text: str) -> str:
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\t', '<|tab|>', text)
        text = unicodedata.normalize('NFKC', text)       
        code_blocks = []
        def replace_code(match):
            code_blocks.append(match.group())
            return f'<|code|>CODE_BLOCK_{len(code_blocks)-1}<|/code|>'       
        text = self.code_pattern.sub(replace_code, text)
        text = self.url_pattern.sub('<URL>', text)
        text = self.email_pattern.sub('<EMAIL>', text)       
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f'<|code|>CODE_BLOCK_{i}<|/code|>', code_block)        
        return text    
    def pre_tokenize(self, text: str) -> List[str]:
        text = self.normalize_text(text)        
        text = re.sub(r'<\|system\|>', ' <system> ', text)
        text = re.sub(r'<\|user\|>', ' <user> ', text)
        text = re.sub(r'<\|assistant\|>', ' <assistant> ', text)
        text = re.sub(r'<\|endoftext\|>', ' <|endoftext|> ', text)       
        tokens = re.findall(r'''
            <[^>]+>|                    # Special tokens
            \b\w+@\w+\.\w+\b|          # Email-like patterns  
            https?://\S+|              # URLs
            ```[\s\S]*?```|            # Code blocks
            `[^`]+`|                   # Inline code
            \b\d+\.?\d*\b|             # Numbers
            \b[a-zA-Z]+(?:'[a-z]*)?|   # Words with optional apostrophes
            [^\w\s]                    # Punctuation
        ''', text, re.VERBOSE)       
        return [token.strip() for token in tokens if token.strip()]    
    def get_pairs(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        pairs = Counter()
        for word, freq in word_freqs.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs   
    def merge_symbols(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        new_word_freqs = {}
        bigram = pair
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs   
    def train_bpe(self, texts: List[str]) -> None:
        print("Training BPE tokenizer...")
        word_freqs = Counter()       
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                print(f"Processing text {i}/{len(texts)}")           
            tokens = self.pre_tokenize(text)
            for token in tokens:
                char_seq = tuple(token)
                if len(char_seq) > 0:
                    word_freqs[char_seq] += 1        
        print(f"Found {len(word_freqs)} unique word patterns")
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= self.min_freq}        
        for term in self.technical_terms:
            if (term,) in word_freqs:
                word_freqs[(term,)] *= 10        
        all_chars = set()
        for word in word_freqs:
            all_chars.update(word)        
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                self.id_to_token[len(self.id_to_token)] = char        
        target_vocab_size = self.vocab_size - len(self.special_tokens)
        num_merges = target_vocab_size - len(self.vocab)       
        for i in range(num_merges):
            if i % 1000 == 0:
                print(f"BPE merge {i}/{num_merges}")            
            pairs = self.get_pairs(word_freqs)
            if not pairs:
                break            
            best_pair = pairs.most_common(1)[0][0]
            word_freqs = self.merge_symbols(best_pair, word_freqs)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.id_to_token[len(self.id_to_token)] = merged_token
            self.bpe_merges.append(best_pair)       
        print(f"BPE training complete. Final vocabulary size: {len(self.vocab)}")
        for word, freq in word_freqs.items():
            for token in word:
                self.token_frequencies[token] += freq  
    def apply_bpe(self, word: str) -> List[str]:
        if word in self.bpe_cache:
            return self.bpe_cache[word]        
        tokens = list(word)
        for merge in self.bpe_merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens = tokens[:i] + [merge[0] + merge[1]] + tokens[i + 2:]
                else:
                    i += 1
        self.bpe_cache[word] = tokens
        return tokens   
    def tokenize(self, text: str) -> List[str]:
        pre_tokens = self.pre_tokenize(text)
        final_tokens = []
        for token in pre_tokens:
            if token in self.special_tokens or token in self.vocab:
                final_tokens.append(token)
            else:
                bpe_tokens = self.apply_bpe(token)
                final_tokens.extend(bpe_tokens)
        return final_tokens   
    def encode_ids(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = ['<bos>'] + tokens + ['<eos>']       
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab['<unk>']))        
        return ids   
    def decode_ids(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for id in ids:
            token = self.id_to_token.get(id, '<unk>')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)        
        text = ''.join(tokens)
        text = text.replace('<|tab|>', '\t')
        text = text.replace('<|newline|>', '\n')
        return text   
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        with open(os.path.join(save_dir, 'merges.txt'), 'w', encoding='utf-8') as f:
            for merge in self.bpe_merges:
                f.write(f"{merge[0]} {merge[1]}\n")
        config = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': self.special_tokens,
            'technical_terms': list(self.technical_terms)
        }
        with open(os.path.join(save_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        with open(os.path.join(save_dir, 'token_frequencies.pkl'), 'wb') as f:
            pickle.dump(dict(self.token_frequencies), f)
        print(f"Tokenizer saved to {save_dir}")    
    def load(self, save_dir: str):
        with open(os.path.join(save_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        with open(os.path.join(save_dir, 'merges.txt'), 'r', encoding='utf-8') as f:
            self.bpe_merges = [tuple(line.strip().split()) for line in f if line.strip()]
        config_file = os.path.join(save_dir, 'tokenizer_config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.vocab_size = config.get('vocab_size', self.vocab_size)
                self.min_freq = config.get('min_freq', self.min_freq)
                if 'technical_terms' in config:
                    self.technical_terms = set(config['technical_terms'])
        freq_file = os.path.join(save_dir, 'token_frequencies.pkl')
        if os.path.exists(freq_file):
            with open(freq_file, 'rb') as f:
                self.token_frequencies = Counter(pickle.load(f))
        self.bpe_cache = {}
        print(f"Tokenizer loaded from {save_dir}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of BPE merges: {len(self.bpe_merges)}")    
    def get_vocab_size(self) -> int:
        return len(self.vocab)   
    def get_token_frequency(self, token: str) -> int:
        return self.token_frequencies.get(token, 0)   
    def analyze_tokenization(self, text: str):
        tokens = self.tokenize(text)
        ids = self.encode_ids(text, add_special_tokens=False)
        print(f"Original text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {ids}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Compression ratio: {len(text.split())/len(tokens):.2f}")
        return tokens, ids
class ConversationDataset:
    """Dataset class for handling conversation data with the custom tokenizer"""  
    def __init__(self, data_file: str, tokenizer: TechnicalTokenizer, max_length: int = 512):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []
        self.load_conversations()   
    def load_conversations(self):
        print(f"Loading conversations from {self.data_file}")       
        if self.data_file.endswith('.jsonl'):
            self.load_jsonl()
        else:
            self.load_text()        
        print(f"Loaded {len(self.conversations)} conversations")   
    def load_jsonl(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    conv = json.loads(line.strip())
                    messages = conv.get("messages", [])
                    if not messages:
                        continue                   
                    text_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "").strip()
                        if not content:
                            continue
                        if role == "system":
                            continue
                        elif role == "user":
                            text_parts.append(f"<user> {content}")
                        elif role == "assistant":
                            text_parts.append(f"<assistant> {content}")
                    
                    if len(text_parts) >= 2:
                        conversation_text = " ".join(text_parts) + " <|endoftext|>"
                        self.conversations.append(conversation_text)               
                except json.JSONDecodeError:
                    continue    
    def load_text(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            content = f.read()
            conversations = content.split('<|endoftext|>\n')
            for conv in conversations:
                conv = conv.strip()
                if conv:
                    self.conversations.append(conv + " <|endoftext|>")   
    def get_tokenized_conversations(self, include_stats=False):
        tokenized = []
        stats = {'total_tokens': 0, 'truncated': 0, 'avg_length': 0}       
        for conv in self.conversations:
            tokens = self.tokenizer.encode_ids(conv)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                stats['truncated'] += 1
            tokenized.append(tokens)
            stats['total_tokens'] += len(tokens)        
        if tokenized:
            stats['avg_length'] = stats['total_tokens'] / len(tokenized)        
        if include_stats:
            return tokenized, stats
        return tokenized    
    def create_training_examples(self, stride: int = None):
        if stride is None:
            stride = self.max_length // 2        
        examples = []
        for conv in self.conversations:
            tokens = self.tokenizer.encode_ids(conv)
            if len(tokens) <= self.max_length:
                examples.append(tokens)
            else:
                for i in range(0, len(tokens), stride):
                    window = tokens[i:i + self.max_length]
                    if len(window) >= 32:
                        examples.append(window)
        return examples
def train_tokenizer_from_files(file_paths: List[str], 
                              vocab_size: int = 32000,
                              min_freq: int = 2,
                              output_dir: str = "tokenizer",
                              max_texts: int = None):
    print(f"Training tokenizer with vocab_size={vocab_size}")
    print(f"Input files: {file_paths}")    
    all_texts = []
    for file_path in file_paths:
        print(f"Loading {file_path}...")        
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        conv = json.loads(line.strip())
                        messages = conv.get("messages", [])
                        text_parts = []
                        for msg in messages:
                            content = msg.get("content", "").strip()
                            if content:
                                text_parts.append(content)
                        if text_parts:
                            all_texts.append(" ".join(text_parts))
                    except json.JSONDecodeError:
                        continue
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = content.split('\n\n')
                for chunk in chunks:
                    if chunk.strip():
                        all_texts.append(chunk.strip())    
    print(f"Loaded {len(all_texts)} texts")
    if max_texts and len(all_texts) > max_texts:
        import random
        random.shuffle(all_texts)
        all_texts = all_texts[:max_texts]
        print(f"Limited to {len(all_texts)} texts")    
    tokenizer = TechnicalTokenizer(vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.train_bpe(all_texts)
    tokenizer.save(output_dir)    
    print("\nTesting tokenization on sample texts:")
    test_texts = [
        "Hello, how can I help you with your Python programming question?",
        "The neural network has 3 hidden layers with ReLU activation functions.",
        "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
        "The derivative of x^2 is 2x, and the integral is (x^3)/3 + C."
    ]    
    for text in test_texts:
        tokenizer.analyze_tokenization(text)
        print()    
    return tokenizer
def main():
    parser = argparse.ArgumentParser(description="Train custom tokenizer for technical content")
    parser.add_argument("--input_files", nargs='+', help="Input text/jsonl files")
    parser.add_argument("--output_dir", default="tokenizer", help="Output directory for tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--max_texts", type=int, help="Maximum number of texts to use for training")
    parser.add_argument("--test_file", help="Test file for analyzing tokenization")
    parser.add_argument("--load_tokenizer", help="Load existing tokenizer from directory")    
    args = parser.parse_args()    
    default_input_file = "/kaggle/input/gpt-based-slm-dataset/slm_training_complete.jsonl"
    default_text_file = "/kaggle/working/text_data/training_data_chat.txt"    
    if not args.input_files and not args.load_tokenizer:
        if os.path.exists(default_input_file):
            args.input_files = [default_input_file]
            print(f"No arguments provided, using default input file: {default_input_file}")
        elif os.path.exists(default_text_file):
            args.input_files = [default_text_file]
            print(f"No arguments provided, using default text file: {default_text_file}")
        else:
            parser.error("No input files or tokenizer directory provided, and default files not found. "
                         "Please specify --input_files or --load_tokenizer.")    
    if args.load_tokenizer:
        tokenizer = TechnicalTokenizer()
        tokenizer.load(args.load_tokenizer)
        if args.test_file:
            print(f"\nTesting on {args.test_file}")
            dataset = ConversationDataset(args.test_file, tokenizer)
            tokenized, stats = dataset.get_tokenized_conversations(include_stats=True)
            print(f"Dataset statistics:")
            print(f"  Total conversations: {len(tokenized)}")
            print(f"  Total tokens: {stats['total_tokens']:,}")
            print(f"  Average tokens per conversation: {stats['avg_length']:.1f}")
            print(f"  Conversations truncated: {stats['truncated']}")
    else:
        tokenizer = train_tokenizer_from_files(
            file_paths=args.input_files,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            output_dir=args.output_dir,
            max_texts=args.max_texts
        )
        if args.test_file:
            print(f"\nTesting on {args.test_file}")
            dataset = ConversationDataset(args.test_file, tokenizer)
            tokenized, stats = dataset.get_tokenized_conversations(include_stats=True)
            print(f"Dataset statistics:")
            print(f"  Total conversations: {len(tokenized)}")
            print(f"  Total tokens: {stats['total_tokens']:,}")
            print(f"  Average tokens per conversation: {stats['avg_length']:.1f}")
            print(f"  Conversations truncated: {stats['truncated']}")

if __name__ == "__main__":
    main()
