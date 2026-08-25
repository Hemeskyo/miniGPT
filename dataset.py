import torch


# Random-window batch sampler.
# Picks `batch_size` random start positions in the token stream and returns:
#   x = a window of `block_size` tokens
#   y = the SAME window shifted by one token (the next-token targets)
# The model learns to predict token i+1 from tokens 0..i at every position.
def get_batch(data, block_size, batch_size, device):
    max_idx = len(data) - block_size - 1
    ix = torch.randint(0, max_idx, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)
