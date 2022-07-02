from multiprocessing.spawn import prepare
import os

import pandas as pd
import random

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class Corpus:
    def __init__(self, file='/crabby/entity/data/ner_dataset.csv'):
        self.training = None
        self.test = None

        directory = os.getcwd()
        df = pd.read_csv(directory + file, encoding="latin1")

        df.columns = df.iloc[0]

        df = df[1:]

        df.columns = ['Sentence #','Word','POS','Tag']
        df = df.rename(columns={"Sentence #": "sentenceNo"})

        df = df.reset_index(drop=True)

        self._extract_sentences(df)


    def get_data(self):
        if not self.training:
            self.prepare_data()

        return self.training, self.test


    def prepare_data(self):
        random.shuffle(self.sentences)
        total_data = len(self.sentences)
        test_set_size = total_data // 10

        self.test = self.sentences[-test_set_size:]
        self.training = self.sentences[:total_data-test_set_size]


    def _extract_sentences(self, df):
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]

        grouped = df.groupby("sentenceNo").apply(agg_func)
        self.sentences = [s for s in grouped]