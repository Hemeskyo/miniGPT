import torch


def get_batch(data, block_size, batch_size, device):
    # data : complete text encoded
    # block_size : length of each extract
    # batch_size : how much extract we handle in same time

    max_idx = len(data) - block_size - 1
    ix = torch.randint(0, max_idx, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)
