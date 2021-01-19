import torch
from torch.utils.data import Dataset


class MaskedLMTweetDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset
        global PAD_ID
        PAD_ID = tokenizer.token_to_id("[PAD]")

    def __getitem__(self, index):
        line = self.dataset[index]
        return {
            'inputs_ids': self.tokenizer.encode(line).ids,
            'attention_mask': []
        }

    def __len__(self):
        return len(self.dataset)


def masked_lm_collate(batch):
    max_len = max(len(el['inputs_ids']) for el in batch)
    res_batch = {'inputs_ids': [], 'attention_mask': []}
    for input_batch in batch:
        vector = input_batch['inputs_ids']
        padded_list = vector + [PAD_ID] * (max_len - len(vector))
        res_batch['inputs_ids'].append(torch.tensor(padded_list))
        res_batch['attention_mask'].append(torch.tensor([1 if token != PAD_ID else 0
                                                         for token in padded_list]))
    res_batch = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else v)
                 for k, v in res_batch.items()}
    return res_batch
