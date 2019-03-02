import torch
from training import training_modules as tr


class training_wrapper:
    def __init__(self,nEpoch,batchSize,model, batchegenerator, size):
        self.size = size
        self.epoch = nEpoch
        self.batch = batchSize
        self.criterion = tr.LabelSmoothing(size=self.size, padding_idx=0, smoothing=0.0)
        self.model_opt = tr.NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        self.model = model
        self.data = batchegenerator

    def train_model(self):
        for y in range(self.epoch):
            print("Epoch" + str(y+1) + ":")
            self.model.train()
            tr.run_epoch(self.data, self.model,tr.SimpleLossCompute(self.model.generator, self.criterion, self.model_opt))
            self.model.eval()
        return self.model







#TODO: What is the ouput of the model? Important for the loss function