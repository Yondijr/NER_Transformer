# TRANSFORMER ARCHITECTURE FOR A NER PROBLEM



from model_build import general_model
from training import training_modules
from preprocessing import dataPreparing
from training import trainer
from training import batching_modules as bm
from validation_testing import validation

trainingData = dataPreparing.data_holder("./data/train.txt")
testData = dataPreparing.data_holder("./data/test.txt", trainingData.srcVocab, trainingData.trgVocab, trainingData.notInVocabSrc )
validData = dataPreparing.data_holder("./data/valid.txt", trainingData.srcVocab, trainingData.trgVocab, trainingData.notInVocabSrc)

model = general_model.make_model(len(trainingData.srcVocab)+1, len(trainingData.trgVocab)+1, N=2)
batchsize = 1
epochs = 1
batchedData = bm.prep_data(trainingData, batchsize)


trainer = trainer.training_wrapper(epochs, batchsize, model, batchedData, len(trainingData.trgVocab) + 1)
# trainedModel = trainer.train_model()


#testing
batchedTesting = bm.prep_data_all(testData)
batchedValid = bm.prep_data_all(validData)

testing = validation.model_tester(model, batchedTesting, batchedValid)
testing.accuracy_test()



