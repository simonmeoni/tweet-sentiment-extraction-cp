import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from tokenizers.models import BPE
from transformers import BertModel

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizerFast
from transformers import BertModel, DistilBertForMaskedLM, DistilBertConfig, EncoderDecoderModel

from src.data.tweet_se_dataset import TweetSentimentExtractionDataset


def init_model(vocabulary_path=None, merges_path=None, configfile=None):
    tokenizer = BPE(
        vocab=vocabulary_path,
        merges=merges_path,
    )

    with open(configfile, "r") as f:
        config = json.load(f)

    model_params = config["model_params"]

    batch_size = model_params["batch_size"]
    dataset = TweetSentimentExtractionDataset(merges_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                             drop_last=True, collate_fn=dataset.collate_function)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Loading models ..")
    vocab_size = model_params["vocab_size"]
    max_length = model_params["max_length"]
    model_config = DistilBertConfig(vocab_size=vocab_size,
                                    max_position_embeddings=max_length,
                                    num_attention_heads=model_params["num_attn_heads"],
                                    num_hidden_layers=model_params["num_hidden_layers"],
                                    hidden_size=model_params["hidden_size"],
                                    type_vocab_size=1,
                                    dropout=0.1,
                                    attention_dropout=0.1,
                                    pad_token_id=0)

    model = DistilBertForMaskedLM(config=model_config)
    model.to(device)
    return model, tokenizer


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def compute_loss(predictions, targets, criterion=None):
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]

    rearranged_output = predictions.view(predictions.shape[0] * predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)

    loss = criterion(rearranged_output, rearranged_target)

    return loss


def train_model(model=None, dataloader=None, optimizer=None, device=None):
    model.train()
    epoch_loss = 0

    for i, (batch) in enumerate(dataloader):
        optimizer.zero_grad()

        batch = batch.to(device)
        out = model(input_ids=batch, attention_mask=batch)
        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, y_targets, criterion)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()


def eval_model(model=None, dataloader=None, optimizer=None, device=None):
    model.eval()
    epoch_loss = 0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        en_input = batch.to(device)

        out = model(input_ids=batch, attention_mask=batch)

        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, y_targets)
        epoch_loss += loss.item()



def main(input_filepath, output_filepath, model, modelparams=None):
    model, tokenizer = init_model()

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=modelparams['lr'])
    criterion = nn.NLLLoss(ignore_index=tokenizer.token_to_id('[PAD]'))

    # MAIN TRAINING LOOP
    for epoch in range(modelparams['num_epochs']):
        print("Starting epoch", epoch + 1)
        # TODO CV here
        train_model()
        eval_model()

    print("Saving model ..")
    #TODO use instead huggingface method
    save_location = modelparams['model_path']
    model_name = modelparams['model_name']
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    save_location = os.path.join(save_location, model_name)
    torch.save(model, save_location)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # pylint: disable=no-value-for-parameter
    main()
