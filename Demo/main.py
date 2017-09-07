from preprocess import PreProcess
#from textMiner import TextMiner
class TextMining():

    def main(self):
        super().__init__()
        folderPath = input("Folder Path:")
        folderPath = "/Users/remosy/DataMining/Demo/textData"
        self.preprocessor = PreProcess(folderPath)
        #self.textminer = TextMiner()
        self.displayData()

    def displayData(self):
        print("Finished")

if __name__ == '__main__':
    TextMining().main()