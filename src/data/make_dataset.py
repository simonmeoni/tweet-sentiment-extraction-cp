# -*- coding: utf-8 -*-
import logging
import re
from concurrent import futures
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('model_filepath', type=click.Path(), default=None)
@click.argument('model', default="custom_model")
def main(input_filepath, output_filepath, model, model_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    train = pd.read_csv(input_filepath + 'train.csv')
    test = pd.read_csv(input_filepath + 'test.csv')
    train = lowercase_and_replace_url(train, 'text')
    train = lowercase_and_replace_url(train, 'selected_text')
    test = lowercase_and_replace_url(test, 'text')
    if model != '':
        tokenizer_dataset_path = output_filepath + '/dataset.txt'
        write_learning_dataset(tokenizer_dataset_path,
                               pd.concat([train, test], ignore_index=True))

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=8000, min_frequency=2,
                             special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                             show_progress=True)
        tokenizer.train(trainer, [tokenizer_dataset_path])
        tokenizer.save(model_filepath + "/tokenizer-" + model + ".json")
    train = extract_span(train)
    train.to_pickle(output_filepath + '/train_processed.pickle')
    test.to_pickle(output_filepath + '/test_processed.pickle')


def extract_span(dataframe):
    dataframe["span"] = ""
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_text = {
            executor.submit(extract_span_thread, df_entry, df_idx):
                df_entry for df_idx, df_entry in enumerate(dataframe.iloc)}
        for future in futures.as_completed(future_text):
            res = future.result()
            dataframe.at[res[0], 'span'] = res[1]

    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def extract_span_thread(df_entry, df_idx):
    start_offset = df_entry['text'].find(df_entry['selected_text'])
    end_offset = start_offset + len(df_entry['selected_text'])
    return df_idx, [start_offset, end_offset]


def write_learning_dataset(path, dataframe):
    with open(path, 'w') as voc_txt:
        for t_entry in dataframe['text']:
            voc_txt.write(t_entry + '\n')
        for sentiment in dataframe["sentiment"].unique().tolist():
            voc_txt.write(sentiment + '\n')


def lowercase_and_replace_url(dataframe, column):
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_text = {executor.submit(lowercase_replace_url_thread, df_entry, df_idx, column):
                       df_entry for df_idx, df_entry in enumerate(dataframe.iloc)}
        for future in futures.as_completed(future_text):
            res = future.result()
            dataframe.at[res[1], column] = res[0]
    dataframe = dataframe[dataframe[column] != '']
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def lowercase_replace_url_thread(entry, df_idx, column):
    text = entry[column].lower().replace("`", "'").strip()
    text = re.sub(r'http[s]?://\S+', '[URL]', text).strip()
    text = text.replace("ï¿½", "'")
    return text, df_idx


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
