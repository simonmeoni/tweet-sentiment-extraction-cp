import torch
from torch.utils.data import Dataset


class MaskedLMTweetDataset(Dataset):
    def __init__(self, txt, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = []
        with open(txt) as f:
            for line in f:
                self.dataset.append(line)

    def __getitem__(self, index):
        line = self.dataset[index]
        return {
            'inputs_ids': line,
            'attention_mask': []
        }


def masked_lm_collate(batch):
    max_len = max(len(el['inputs_ids']) for el in batch)
    res_batch = {}
    for elem in batch:
            for key, value in elem.items():
                padded_list = [0] * (max_len - len(value))
                padded_vector = torch.cat((value, torch.LongTensor(padded_list)))
                res_batch[key].append(padded_vector)
    res_batch = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else v)
                 for k, v in res_batch.items()}
    return res_batch
