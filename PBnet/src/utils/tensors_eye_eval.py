import torch


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):

    posbatch = [b[1] for b in batch]
    audiobatch = [b[0] for b in batch]
    eyebatch = [b[2] for b in batch]
    lenbatch = [len(b[0]) for b in batch]
    startbatch = [b[4] for b in batch]
    videonamebatch=[b[3] for b in batch]
    poseyebatch=[b[5] for b in batch]

    posbatchTensor = collate_tensors(posbatch)
    audiobatchTensor = collate_tensors(audiobatch)
    eyebatchTensor = collate_tensors(eyebatch)
    poseyebatchTensor = collate_tensors(poseyebatch)
    # startbatchTensor = collate_tensors(startbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    batch = {"x":poseyebatchTensor,"p": posbatchTensor, "y": audiobatchTensor,
             "e": eyebatchTensor, "mask": maskbatchTensor, "lengths": lenbatchTensor,
             "videoname": videonamebatch, "start": startbatch}
    return batch
