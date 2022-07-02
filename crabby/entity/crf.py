from crabby.entity.data_preparation import Corpus
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
import pickle
import os

class CRF_Model():
    def __init__(self, shouldTrain=True):
        if shouldTrain:
            self._train_model_from_scratch()
        else:
            self._load_model()


    def _load_model(self):
        self.crf = pickle.load(open(os.getcwd() + "/crabby/entity/model/crf", 'rb'))


    def _train_model_from_scratch(self):
        corpus = Corpus()

        self.training, self.test = corpus.get_data()
        self.crf = CRF(algorithm='lbfgs',
                       c1=0.1,
                       c2=0.1,
                       max_iterations=100,
                       all_possible_transitions=True)

        self.train()
        self.evaluate()


    def predict(self, sentence):
        features = CRF_Model.sent2features(sentence)
        return self.crf.predict([features])


    def train(self):
        X_train = [CRF_Model.sent2features(s) for s in self.training]
        y_train = [CRF_Model.sent2labels(s) for s in self.training]

        self.crf.fit(X_train, y_train)

    def evaluate(self):
        labels = list(self.crf.classes_)
        labels.remove('O')

        X_test = [CRF_Model.sent2features(s) for s in self.test]
        y_test = [CRF_Model.sent2labels(s) for s in self.test]

        y_pred = self.crf.predict(X_test)
        score = metrics.flat_f1_score(y_test, y_pred,
                                      average='weighted', labels=labels)
        print(score)


    def save_model(self):
        pickle.dump(self.crf, open(os.getcwd() + "/crabby/entity/model/crf", 'wb'))


    @staticmethod
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        return features


    @staticmethod
    def sent2features(sent):
        return [CRF_Model.word2features(sent, i) for i in range(len(sent))]

    @staticmethod
    def sent2labels(sent):
        return [label for _, _, label in sent]
