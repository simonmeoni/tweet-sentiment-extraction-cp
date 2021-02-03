import gc
import logging
import os
from datetime import datetime
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy import mean
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from transformers import DistilBertForMaskedLM, DistilBertConfig
from src.data.masked_lm_tse_dataset import MaskedLMTweetDataset
from src.models.distillbert_tokens_classification_train import count_parameters
import wandb

# pylint: disable=too-many-arguments, too-many-locals

LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)


def init_model(tokenizer, config, device):
    logger.info("🍌 Loading model...")
    max_length = config["max_length"]
    model_parameter_path = DistilBertConfig(vocab_size=tokenizer.get_vocab_size(),
                                            max_position_embeddings=max_length,
                                            num_attn_heads=config["num_attn_heads"],
                                            n_layers=config["n_layers"],
                                            hidden_dim=config["hidden_dim"],
                                            type_vocab_size=1,
                                            dropout=0.1,
                                            attention_dropout=0.1,
                                            pad_token_id=0)
    model = DistilBertForMaskedLM(config=model_parameter_path)
    model.to(device)
    logger.info(f'The model has {count_parameters(model):} trainable parameters')
    return model


def init_tokenizer(tokenizer_path):
    logger.info("🍌 Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def compute_loss(predictions, targets, criterion=None):
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]
    rearranged_output = predictions.view(predictions.shape[0] * predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)
    loss = criterion(rearranged_output, rearranged_target)

    return loss


def train_model(model, dataloader, optimizer, criterion, device, id_fold):
    model.train()
    epoch_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        out = model(input_ids=batch['inputs_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
        prediction_scores = out.logits
        predictions = f.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, batch['inputs_ids'], criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        wandb.log({"loss {}".format(id_fold): loss.item()})
        epoch_loss.append(loss.item())
    logger.info('train epoch loss : {}'.format(mean(epoch_loss)))


def eval_model(model, dataloader, optimizer, criterion, device, id_fold):
    model.eval()
    epoch_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        out = model(input_ids=batch['inputs_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
        prediction_scores = out.logits
        predictions = f.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, batch['inputs_ids'], criterion)
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

    txt_dataset = []
    with open(hyperparameter_defaults['dataset_path']) as file:
        for line in file:
            txt_dataset.append(line[:-1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device : {}".format(device))
    tokenizer = init_tokenizer(hyperparameter_defaults['tokenizer_path'])
    criterion = nn.NLLLoss(ignore_index=tokenizer.token_to_id('[PAD]'))
    folds = KFold(n_splits=hyperparameter_defaults['folds'], shuffle=False)
    cv_score = []
    for id_fold, fold in enumerate(folds.split(txt_dataset)):
        logger.info('beginning fold n°{}'.format(id_fold + 1))
        model = init_model(tokenizer, hyperparameter_defaults, device)
        optimizer = optim.Adam(model.parameters(), lr=hyperparameter_defaults['lr'])

        train_fold, eval_fold = fold
        train_fold = [txt_dataset[idx_train] for idx_train in train_fold]
        eval_fold = [txt_dataset[idx_eval] for idx_eval in eval_fold]
        train_dataset = MaskedLMTweetDataset(train_fold, tokenizer)
        eval_dataset = MaskedLMTweetDataset(eval_fold, tokenizer)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=hyperparameter_defaults["batch_size"],
                                      shuffle=False, drop_last=True,
                                      collate_fn=train_dataset.masked_lm_collate)
        eval_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=hyperparameter_defaults["batch_size"],
                                     shuffle=False, drop_last=True,
                                     collate_fn=eval_dataset.masked_lm_collate)

        wandb.watch(model)
        for epoch in range(hyperparameter_defaults['num_epochs']):
            logger.info("Starting epoch {}".format(epoch + 1))
            train_model(model, train_dataloader, optimizer, criterion, device, id_fold)
        logger.info("Saving model ..")
        current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_location = hyperparameter_defaults['model_path']
        model_name = hyperparameter_defaults['model_name'] + '-' + current_datetime + \
                     '-fold-{}'.format(id_fold + 1)
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        save_location = os.path.join(save_location, model_name)
        torch.save(model, save_location)
        wandb.save(save_location)
        score = eval_model(model, eval_dataloader, optimizer, criterion, device, id_fold)
        del model
        gc.collect()
        cv_score.append(score)
        wandb.log({"cv-score": score})
    logger.info('CV score : {}'.format(cv_score))


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # pylint: disable=no-value-for-parameter
    main()
