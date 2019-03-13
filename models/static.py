import torch

FILE_MODEL = './resources/out/model.pkl'
FILE_LOG = './resources/out/log1.txt'
RESOURCES_DATA = './resources/data/'
RESOURCES_DATA_TEST = './resources/test/'
RESOURCES_DATA_TRAIN = './resources/train/'
# FILE_WORD_2_VEC = './resources/embedding/word2vec_vi.txt'
FILE_WORD_2_VEC = './resources/embedding/vocab_train_1'
FILE_CHAR_2_VEC = './resources/embedding/char2vec'
FILE_TAG_2_INT = './resources/embedding/tags'
VOCAB_ONE_HOT = './resources/embedding/vocab_train_1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH_SENTENCE = 100
MAX_LENGTH_WORD = 50
UNKNOWN = '</s>'
NUMBER = "<number>"
