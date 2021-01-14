# -*- coding: utf-8 -*-
from concurrent import futures

import click
import logging
import pandas as pd
import re
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer


from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('model', default="from_scratch")
def main(input_filepath, output_filepath, model):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    train = pd.read_csv(input_filepath + 'train.csv')
    test = pd.read_csv(input_filepath + 'test.csv')
    train = lowercase_and_replace_url(train)
    test = lowercase_and_replace_url(test)
    if model == 'from_scratch':
        write_learning_dataset(output_filepath + '/tokenizer_dataset.txt',
                               pd.concat([train, test], ignore_index=True))

        tokenizer = Tokenizer(models.BPE())
        trainer = BpeTrainer(vocab_size = 52_000, min_frequency = 2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=True)
        tokenizer.train([output_filepath + '/tokenizer_dataset.txt'], trainer=trainer)
        tokenizer.save(output_filepath + "/from_scratch-tokenizer.json")


def write_learning_dataset(path, dataframe):
    with open(path, 'w') as voc_txt:
        for t_entry in dataframe['text']:
            voc_txt.write(t_entry + '\n')
        for sentiment in dataframe["sentiment"].unique().tolist():
            voc_txt.write(sentiment + '\n')
    

def lowercase_and_replace_url(dataframe):
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)
    column = 'text'
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
    return text, df_idx


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
