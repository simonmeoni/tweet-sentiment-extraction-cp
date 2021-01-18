import torch
from torch.utils.data import Dataset


class TweetSentimentExtractionDataset(Dataset):
    def __init__(self, dataset, sentence_piece):
        self.dataset = dataset
        self.st_voc = []
        self.sentence_piece = sentence_piece
        self.max_seq_len = 0
        self.__init_sentiment_vocab()

    def __init_sentiment_vocab(self):
        self.st_voc = [*self.dataset['sentiment'].unique()]

    def get_st_vocab(self, sentiment_id):
        return self.sentence_piece.EncodeAsIds(self.st_voc[sentiment_id])[0]

    def get_vocab_size(self):
        return self.sentence_piece.vocab_size() + 1

    @staticmethod
    def get_sentence_vocab_size():
        # [CLS, SEQ, PAD, UNK, S1, S2]
        return 6

    def __getitem__(self, index):
        words_emb, sentence_emb = self.vectorize(self.dataset.iloc[index]['text'],
                                                 self.dataset.iloc[index]['sentiment'])
        selected_vector = self.fill_selected_vector(sentence_emb,
                                                    self.dataset.iloc[index]['selected_vector']) \
            if 'selected_vector' in self.dataset.columns else 'no selected vector'
        return {
            'words_embedding': words_emb,
            'sentence_embedding': sentence_emb,
            'sentiment_i': self.get_sentiment_i(self.dataset.iloc[index]['sentiment']),
            'selected_text': self.dataset.iloc[index]['selected_text']
            if 'selected_text' in self.dataset.columns else 'no selected text',
            'selected_vector': selected_vector
        }

    def __len__(self):
        return len(self.dataset)

    # noinspection PyArgumentList
    def vectorize(self, tokens, sentiment):
        st_vector = self.get_st_vocab(self.get_sentiment_i(sentiment))
        vector = self.sentence_piece.EncodeAsIds(tokens)
        word_embedding = [self.get_cls()] + [st_vector] + [self.get_sep()] + \
                         vector + [self.get_sep()]
        padding = [self.get_pad()] * (self.max_seq_len - len(word_embedding) + 5)
        sentence_embedding = [self.get_cls()] + [4] + \
                             [self.get_sep()] + [5] * len(vector) + [self.get_sep()]

        return torch.LongTensor(word_embedding + padding), \
               torch.LongTensor(sentence_embedding + padding)

    def get_mask(self):
        return self.sentence_piece.vocab_size()

    def get_pad(self):
        return self.sentence_piece.pad_id()

    def get_cls(self):
        return self.sentence_piece.bos_id()

    def get_sep(self):
        return self.sentence_piece.eos_id()

    def get_tokens(self, ids):
        return ' '.join([self.sentence_piece.Decode(i) if i != self.get_mask()
                         else MASK for i in ids.tolist()]).strip()

    def get_sentiment_i(self, st_token):
        return self.st_voc.index(st_token)

    @staticmethod
    def fill_selected_vector(sentence_emb, selected_vector):
        selected_vector_padded = selected_vector
        sep_index = ((sentence_emb == 3).nonzero(as_tuple=False)).squeeze().tolist()
        selected_vector_padded = [0] * len(sentence_emb[:sep_index[0] + 1]) + \
                                 selected_vector_padded + [0] * len(sentence_emb[sep_index[1]:])
        return torch.LongTensor(selected_vector_padded)
