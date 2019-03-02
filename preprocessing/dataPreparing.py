# Data Loading

class data_holder:
    def __init__(self,path, srcvocabulary=False, trgvocabulary=False, dummySrc = False):
        self.presrcvocabulary = srcvocabulary
        self.pretrgvocabulary = trgvocabulary
        self.src = path
        self.rawdata = (open(self.src, "r+"))
        self.rawdata = self.rawdata.read()
        self.rawdata = self.rawdata.splitlines()
        self.data, self.label = self.format_input()
        if not self.presrcvocabulary and not self.pretrgvocabulary:
            self.srcVocab = self.build_vocab(self.data)
            self.trgVocab = self.build_vocab(self.label)
            self.notInVocabSrc = len(set(self.data)) + 1
            self.notInVocabTrg = len(set(self.label)) + 1
        else:
            self.srcVocab = srcvocabulary
            self.trgVocab = trgvocabulary
            self.notInVocabSrc = dummySrc

        self.dataTokenized, self.labelTokenized = self.tokenize() #!!
        print(self.dataTokenized)

    def format_input(self):
        data = []
        label =[]
        for x in range(0, len(self.rawdata)):
            self.rawdata[x] = self.rawdata[x].split(" ")
            if len(self.rawdata[x]) == 1:
                self.rawdata[x] = ["!End!","","","!End!"]
        for x in range(0, len(self.rawdata)):
            data.append(self.rawdata[x][0])
            label.append(self.rawdata[x][3])
        return data, label

    def build_vocab(self, totranslate):
        trainWord_to_ix = {word: i for i, word in enumerate(set(totranslate))}
        return trainWord_to_ix

    def tokenize(self):
        tokenizedData = []  # !
        tokenizedLabel = []
        singleSentence = []
        singleSentenceLabel = []
        for x in range(0, len(self.data)):
            if self.data[x] != "!End!":
                if self.data[x] in self.srcVocab:
                    singleSentence.append(self.srcVocab[self.data[x]])
                    singleSentenceLabel.append(self.trgVocab[self.label[x]])
                else:
                    singleSentence.append(self.notInVocabSrc)
                    singleSentenceLabel.append(self.trgVocab[self.label[x]])
            else:
                tokenizedData.append(singleSentence)
                tokenizedLabel.append(singleSentenceLabel)
                singleSentence = []
                singleSentenceLabel = []
        return tokenizedData,tokenizedLabel


#Todo:  TOkenize of labeling often has an empty line at the start. Might effect the network