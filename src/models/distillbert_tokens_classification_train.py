import logging
import os
from datetime import datetime
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy import mean
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

import wandb

# pylint: disable=too-many-arguments, too-many-locals
from src.data.token_classification_tse_dataset import TokenClassificationTweetDataset


def init_model(model_name, device):
    print("🍌 Loading model...")
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model


def init_tokenizer(model_name):
    print("🍌 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, )
    return tokenizer


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def compute_loss(predictions, targets, criterion=None):
    return criterion(predictions.transpose(1, 2).reshape(-1, 2), targets.reshape(-1))


class FineTuningClassifier(nn.Module):
    def __init__(self, zn_size):
        super().__init__()
        self.conv1d = nn.Conv1d(zn_size, 2, kernel_size=3, padding=1)

    def forward(self, z_n):
        return self.conv1d(z_n)


def train_model(model, classifier, dataloader, optimizer, criterion, device, id_fold):
    model.train()
    classifier.train()
    epoch_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        out = model(input_ids=batch['x_obs'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
        prediction_scores = classifier(out.last_hidden_state.transpose(1, 2))
        predictions = f.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, batch['target'], criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        wandb.log({"loss {}".format(id_fold): loss.item()})
        epoch_loss.append(loss.item())
    print('train epoch loss : {}'.format(mean(epoch_loss)))


def eval_model(model, classifier, dataloader, optimizer, criterion, device, id_fold):
    model.eval()
    classifier.eval()
    epoch_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        out = model(input_ids=batch['x_obs'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
        prediction_scores = classifier(out.last_hidden_state.transpose(1, 2))
        predictions = f.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, batch['target'], criterion)
        epoch_loss.append(loss.item())
        wandb.log({"CV loss {}".format(id_fold): mean(loss.item())})
        break
    return mean(epoch_loss)


@click.command()
@click.option('--learning_rate')
@click.option('--batch_size')
@click.option('--max_length')
@click.option('--num_attn_heads')
@click.option('--n_layers')
@click.option('--hidden_dim')
@click.option('--num_epochs')
@click.option('--folds')
@click.option('--model_name')
@click.option('--model_path')
@click.option('--tokenizer_path')
@click.option('--dataset_path')
def main(learning_rate,
         batch_size,
         max_length,
         num_attn_heads,
         n_layers,
         hidden_dim,
         num_epochs,
         folds,
         model_name,
         model_path,
         tokenizer_path,
         dataset_path):
    hyperparameter_defaults = dict(
        lr=float(learning_rate),
        batch_size=int(batch_size),
        max_length=int(max_length),
        num_attn_heads=int(num_attn_heads),
        n_layers=int(n_layers),
        hidden_dim=int(hidden_dim),
        num_epochs=int(num_epochs),
        folds=int(folds),
        model_name=model_name,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        dataset_path=dataset_path,
    )
    wandb.init(project="tweet-se-competition", config=hyperparameter_defaults)
    dataset = pd.read_pickle(dataset_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("Using tokenizer from model : ", model_name)
    tokenizer = init_tokenizer(hyperparameter_defaults['model_name'])
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
    folds = KFold(n_splits=hyperparameter_defaults['folds'], shuffle=False)
    cv_score = []
    for id_fold, fold in enumerate(folds.split(dataset[dataset['sentiment'] != 'neutral'])):
        print('beginning fold n°{}'.format(id_fold + 1))
        model = init_model(hyperparameter_defaults['model_name'], device)
        classifier = FineTuningClassifier(model.config.dim)
        optimizer = optim.Adam(model.parameters(), lr=hyperparameter_defaults['lr'])
        train_fold, eval_fold = fold
        train_fold = dataset.iloc[train_fold]
        eval_fold = dataset.iloc[eval_fold]
        train_dataset = TokenClassificationTweetDataset(train_fold, tokenizer)
        eval_dataset = TokenClassificationTweetDataset(eval_fold, tokenizer)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=hyperparameter_defaults["batch_size"],
                                      shuffle=False, drop_last=True,
                                      collate_fn=train_dataset.token_classification_collate)
        eval_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=hyperparameter_defaults["batch_size"],
                                     shuffle=False, drop_last=True,
                                     collate_fn=eval_dataset.token_classification_collate)

        wandb.watch(model)
        for epoch in range(hyperparameter_defaults['num_epochs']):
            print("Starting epoch", epoch + 1)
            train_model(model, classifier, train_dataloader, optimizer, criterion, device, id_fold)
        print("Saving model ..")
        current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_location = hyperparameter_defaults['model_path']
        model_name = hyperparameter_defaults['model_name'] + '-' + current_datetime + \
                     '-fold-{}'.format(id_fold + 1)
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        save_location = os.path.join(save_location, model_name)
        torch.save({'model': model, 'classfier': classifier}, save_location)
        wandb.save(save_location)
        score = eval_model(model, classifier, eval_dataloader, optimizer, criterion, device,
                           id_fold)
        cv_score.append(score)
        wandb.log({"cv-score": score})
    print('CV score : {}'.format(cv_score))


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
