##    Configuaration  ##
batch_size = 8
num_classes = 121
dropout = 0.5
learning_rate = 0.001
momentum = 0.9
epoch = 100
vocab = 122

save_path = "./resluts"
train_corpus = "./data/train_corpus.csv"
valid_corpus = "./data/test_corpus.csv"

os.makedirs(save_path, exist_ok=True)
