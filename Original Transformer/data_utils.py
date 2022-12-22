from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

from tqdm import tqdm

class English2Spainsh_Dataset(Dataset):

    def __init__(self, spanish_path, english_path):

        english_file = open(english_path, "r")
        spanish_file = open(spanish_path, "r")

        self.spanish_tokenizer = get_tokenizer('spacy', language="es_core_news_sm")
        self.english_tokenizer = get_tokenizer('spacy', language="en_core_web_sm")

        counter_spanish, counter_english = Counter(), Counter()
        self.spanish_dataset, self.english_dataset = [], []

        print("Building English and Spanish Vocabularies")
        for spa, eng in tqdm(zip(spanish_file, english_file)):
            
            spa = spa.lower()
            eng = eng.lower()
            
            self.spanish_dataset.append(spa)
            self.english_dataset.append(eng)

            counter_spanish.update(self.spanish_tokenizer(spa))
            counter_english.update(self.english_tokenizer(eng))
    
        self.english_vocab = vocab(counter_english, min_freq=10, specials=('<UNK>', '<BOS>', '<EOS>', '<PAD>'))
        self.spanish_vocab = vocab(counter_spanish, min_freq=10, specials=('<UNK>', '<BOS>', '<EOS>', '<PAD>'))

        self.text_en_transform = lambda x: [self.english_vocab['<BOS>']] + [self.english_vocab[token] for token in self.english_tokenizer(x)] + [self.english_vocab['<EOS>']]
        self.text_es_transform = lambda x: [self.spanish_vocab['<BOS>']] + [self.spanish_vocab[token] for token in self.spanish_tokenizer(x)] + [self.spanish_vocab['<EOS>']]
        
    def __len__(self):
        return len(self.spanish_dataset)

    def __getitem__(self, ix):
        return self.english_dataset[ix], self.spanish_dataset[ix]

    def collate_fn(self, batch):
        english_list, spanish_list = [], []

        for (eng, spa) in batch:
            english_list.append(torch.tensor(self.text_en_transform(eng)))
            spanish_list.append(torch.tensor(self.text_es_transform(spa)))
        
        return pad_sequence(english_list, padding_value=float(self.english_vocab(['<PAD>'])[0])), pad_sequence(spanish_list, padding_value=float(self.spanish_vocab(['<PAD>'])[0]))        