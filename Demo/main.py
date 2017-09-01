from preprocess import PreProcess
from textMiner import TextMiner
class TextMining():

    def __init__(self):
        super().__init__()
        self.preprocessor = PreProcess()
        self.textminer = TextMiner()
        self.displayData()

    def displayData(self):
        print("Finished")
