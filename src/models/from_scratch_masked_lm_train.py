import json
import logging
import os
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from dotenv import find_dotenv, load_dotenv
from tokenizers.models import BPE
from transformers import DistilBertForMaskedLM, DistilBertConfig
from torch.utils.data import DataLoader
from src.data.masked_lm_tse_dataset import MaskedLMTweetDataset, masked_lm_collate


def init_model(tokenizer_path, dataset_path, model_parameter_path, device):
    tokenizer = BPE(
        vocab=tokenizer_path,
        merges=tokenizer_path,
    )
    with open(model_parameter_path, "r") as file:
        config = json.load(file)
    batch_size = config["batch_size"]
    dataset = MaskedLMTweetDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                             drop_last=True, collate_fn=masked_lm_collate)
    print("Loading models ..")
    vocab_size = config["vocab_size"]
    max_length = config["max_length"]
    model_parameter_path = DistilBertConfig(vocab_size=vocab_size,
                                            max_position_embeddings=max_length,
                                            num_attention_heads=config["num_attn_heads"],
                                            num_hidden_layers=config["num_hidden_layers"],
                                            hidden_size=config["hidden_size"],
                                            type_vocab_size=1,
                                            dropout=0.1,
                                            attention_dropout=0.1,
                                            pad_token_id=0)
    model = DistilBertForMaskedLM(config=model_parameter_path)
    model.to(device)
    return model, tokenizer, dataloader


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def compute_loss(predictions, targets, criterion=None):
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]
    rearranged_output = predictions.view(predictions.shape[0] * predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)
    loss = criterion(rearranged_output, rearranged_target)

    return loss


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(input_ids=batch['inputs_ids'], attention_mask=batch['attention_mask'])
        prediction_scores = out[1]
        predictions = f.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, batch['inputs_ids'], criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()


def eval_model(model, dataloader, optimizer, device):
    model.eval()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(input_ids=batch['inputs_ids'], attention_mask=batch['attention_mask'])
        prediction_scores = out[1]
        predictions = f.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, batch['inputs_ids'])
        epoch_loss += loss.item()


@click.command()
@click.argument('tokenizer_path', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('model_parameters', type=click.Path(exists=True))
def main(tokenizer_path, dataset_path, model_parameters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model, tokenizer, dataloader = init_model(tokenizer_path, dataset_path,
                                              model_parameters, device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=model_parameters['lr'])
    criterion = nn.NLLLoss(ignore_index=tokenizer.token_to_id('[PAD]'))

    # MAIN TRAINING LOOP
    for epoch in range(model_parameters['num_epochs']):
        print("Starting epoch", epoch + 1)
        # TODO CV here
        train_model(model, dataloader, optimizer, criterion, device)
        eval_model(model, dataloader, optimizer, device)

    print("Saving model ..")
    save_location = model_parameters['model_path']
    model_name = model_parameters['model_name']
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
