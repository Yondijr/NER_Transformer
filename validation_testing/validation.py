import torch
from torch.autograd import Variable

from validation_testing import validation_modules as vm

class model_tester:
    def __init__(self,model,testing,validation):
        self.model = model
        self.testingData = testing
        self.validData = validation

    def accuracy_test(self):

        # Calculate Accuracy for the test data
        self.model.eval()
        hits = 0.0
        comparisons = 0.0

        for x in range(0, 10):
            print(x)
            src = Variable(torch.LongTensor(self.testingData.src[x].unsqueeze(0)))
            src_mask = Variable(torch.ones(1, 1, len(self.testingData.src[x])))
            initial = self.testingData.trg[x]
            decode = vm.greedy_decode(self.model, src, src_mask, max_len=len(self.testingData.src[x]))
            for y in range(0, len(initial)):
                comparisons += 1
                start = int(initial.numpy()[y])
                end = decode.numpy()[0][y]
                if start == end:
                    hits += 1
        print("For the test data " + str(hits / comparisons) + "% accuray were achieved")


        #Todo: implement function to calculate accuracy of the model for the test data. See below

    def example_predictor(self,toTranslate):
        self.model.eval()
        src = Variable(torch.LongTensor(self.validData.src[toTranslate].unsqueeze(0)))
        src_mask = Variable(torch.ones(1, 1, len(self.validData.src[toTranslate])))
        decode = vm.greedy_decode(self.model, src, src_mask, max_len=len(self.validData.src[toTranslate]))
        print(self.validData.trg[toTranslate])
        print(decode)





