import torch
import torch.nn as nn
from torch.nn import ReLU, functional as F

# hyperparameters
# ------------
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
epochs = 10000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------


with open('gita.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # Linear transformation for key vectors
        self.keys = nn.Linear(n_embd, head_size, bias = False)
        # Linear transformation for query vectors
        self.query = nn.Linear(n_embd, head_size, bias = False)
        # Linear transformation for value vectors
        self.value = nn.Linear(n_embd, head_size, bias = False)
        # Create lower triangular matrix for masked attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Compute key, value, query vectors from input
        B,T,C = x.shape
        k = self.keys(x) #  B,T,C -> Batch, Time(sequence length), Channels
        v = self.value(x)
        q = self.query(x)

        # Compute attention scores/weights
        wei = q @ k.transpose(-2,-1) # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        # Mask future positions
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
 
        # Apply dropout for regularization
        wei = self.dropout(wei)
        # Compute weighted sum of values
        out = wei @ v
        return out

# multi head attention
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create multiple attention heads in parallel using ModuleList
        self.attn = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        # Project concatenated outputs back to embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all attention heads
        out = torch.cat([attn(x) for attn in self.attn], dim=-1)
        # Project back to embedding dimension
        out = self.proj(out)
        # Apply dropout
        out = self.dropout(out)
        return out


# feed forward
class FeedForwardLayer(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffwd(x)

# transfromer block
class Block(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForwardLayer(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x # B, T, C

# gpt
class GPT(nn.Module):

    def __init__(self, num_heads, head_size, num_layers):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, n_embd)
        self.posembd = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(num_heads, head_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):  # Add targets argument with default None
        embd = self.embd(x)  # (B,T,C)
        posembd = self.posembd(torch.arange(x.shape[1], device=x.device))  # Pass token positions
        o = embd + posembd
        out = self.blocks(o)  # (B ,T ,C)
        out = self.ln_f(out)  # (B, T, C)
        logits = self.lm_head(out)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # Call forward with idx_cond
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPT(num_heads=n_head,head_size=n_embd // n_head,num_layers = n_layer)
m = model.to(device)


print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
