import logging
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)


class TokenClassificationTweetDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = self.init_dataset(dataset)
        self.pad_id = tokenizer.pad_token_id

    def __getitem__(self, index):
        entry = self.dataset.loc[index]
        return {
            'x_obs': entry['tokens'],
            'attention_mask': [],
            'target': entry['selected_tokens'],
            'text': entry['text'],
            'selected_text': entry['selected_text']
        }

    def __len__(self):
        return len(self.dataset)

    def token_classification_collate(self, batch):
        max_len = max(len(el['x_obs']) for el in batch)
        res_batch = {'x_obs': [],
                     'attention_mask': [],
                     'target': [],
                     'selected_text': [],
                     'text': [],
                     }
        for input_batch in batch:
            vector = input_batch['x_obs']
            padded_list = [self.pad_id] * (max_len - len(vector))
            x_obs = input_batch['x_obs'] + padded_list
            res_batch['x_obs'].append(torch.LongTensor(x_obs))
            res_batch['target'].append(torch.LongTensor(input_batch['target'] + padded_list))
            res_batch['attention_mask'].append(torch.LongTensor([1 if token != self.pad_id else 0
                                                                 for token in x_obs]))
            res_batch['text'].append(input_batch['text'])
            res_batch['selected_text'].append(input_batch['selected_text'])
        res_batch = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else v)
                     for k, v in res_batch.items()}
        return res_batch

    def init_dataset(self, dataset):
        logger.info('Initialize Dataset ....')
        new_dataset = pd.concat([dataset,
                                 pd.DataFrame(columns=["selected_tokens", "tokens"])])
        for df_idx, df_entry in tqdm(new_dataset.iterrows(),
                                     total=new_dataset.shape[0],
                                     desc="span token extractions: "):
            res = self.extract_span_tokens(df_entry, df_idx)
            new_dataset.at[res[0], "tokens"] = res[1]
            new_dataset.at[res[0], "selected_tokens"] = res[2]

        new_dataset = new_dataset.reset_index(drop=True)

        return new_dataset

    def extract_span_tokens(self, df_entry, df_idx):
        text = df_entry["text"]
        tokens = self.tokenizer(text, return_offsets_mapping=True)
        ids = tokens['input_ids'][1:-1]
        selected_tokens = [0] * len(ids)
        end_idx, start_idx = self.get_span_token_index(
            df_entry["span"][1],
            tokens['offset_mapping'][1:-1],
            df_entry["span"][0])
        selected_tokens[start_idx:end_idx + 1] = [0] + [1] * \
                                                 len(selected_tokens[start_idx:end_idx + 1]) + [0]
        return df_idx, tokens['input_ids'], selected_tokens

    @staticmethod
    def get_span_token_index(end, offsets, start):
        start_idx = 0
        end_idx = 0
        for idx_offset, offset in enumerate(offsets):
            start_token = offset[0]
            end_token = offset[1]
            if start_token <= start <= end_token:
                start_idx = idx_offset
            if start_token <= end <= end_token:
                end_idx = idx_offset
                break
        return end_idx, start_idx
