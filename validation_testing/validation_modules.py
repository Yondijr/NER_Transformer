import torch
from torch.autograd import Variable

from model_build import decoder

def greedy_decode(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(decoder.subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys