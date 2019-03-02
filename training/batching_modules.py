import torch
from torch.autograd import Variable
from model_build import decoder

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_y = self.trg_y.type(torch.FloatTensor)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            decoder.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# Generator for batches that hold all Data
def prep_data(trainingData, batchSize):
    maxLength = 0
    for x in trainingData.dataTokenized:
        if len(x) > maxLength:
            maxLength = len(x)
    for x in range(0, len(trainingData.dataTokenized)):
        if len(trainingData.dataTokenized[x]) < maxLength:
            for y in range(0, (maxLength - len(trainingData.dataTokenized[x]))):
                trainingData.dataTokenized[x].append(trainingData.srcVocab['!End!'])
                trainingData.labelTokenized[x].append(trainingData.trgVocab['!End!'])

    numberOfBatches = len(trainingData.dataTokenized) // batchSize + 1
    if len(trainingData.dataTokenized) % batchSize == 0:
        numberOfBatches = len(trainingData.dataTokenized) // batchSize

    for x in range(0, numberOfBatches):
        specificSrc = trainingData.dataTokenized[x * batchSize:x * batchSize + batchSize ]
        specificTrg = trainingData.labelTokenized[x * batchSize:x * batchSize + batchSize]
        convertSrc = torch.LongTensor(specificSrc)
        convertTrg = torch.LongTensor(specificTrg)
        srcOut = Variable(convertSrc, requires_grad=False)
        trgOut = Variable(convertTrg, requires_grad=False)
        yield Batch(srcOut, trgOut, 0)


# Alternative only one batch ( for testing and validating)
def prep_data_all(trainingData,):
    maxLength = 0
    for x in trainingData.dataTokenized:
        if len(x) > maxLength:
            maxLength = len(x)
    for x in range(0, len(trainingData.dataTokenized)):
        if len(trainingData.dataTokenized[x]) < maxLength:
            for y in range(0, (maxLength - len(trainingData.dataTokenized[x]))):
                trainingData.dataTokenized[x].append(trainingData.srcVocab['!End!'])
                trainingData.labelTokenized[x].append(trainingData.trgVocab['!End!'])

    srcOut = torch.LongTensor(trainingData.dataTokenized)
    trgOut = torch.LongTensor(trainingData.labelTokenized)
    return Batch(srcOut, trgOut, 0)

