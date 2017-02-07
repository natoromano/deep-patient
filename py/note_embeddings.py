"""
Fits embedding models to clinical notes.

Methods:
    - topic model (LDA)
"""

from itertools import islice
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
import cPickle
import linecache

import sklearn
from sklearn.decomposition import LatentDirichletAllocation

NOTES_PATH = '/scratch/users/naromano/deep-patient/shah/notes.csv'
MAX_TERM = 7299366



# Utility functions

def get_next_lines(data_file, start, stop=None):
    """
    Returns a generator with lines in file.

    Note: indices must be slide.
    """
    if stop is None:
        return (line.strip() for line in islice(data_file, start))
    else:
        return (line.strip() for line in islice(data_file, start, stop))


def parse_line(line):
    """
    Returns:
        patient_id, note_id, note_year, list of terms
    """
    patient_id, note_id, note_year, terms = line.split("\t")
    return patient_id, note_id, note_year, terms.split(" ")



# Topic Modeling

class TopicModel(LatentDirichletAllocation):
    """
    Wrapper around sklearn's LDA model to handle dumped data.
    """

    VOCAB_PATH = "../data/terms.txt"


    def fit(self, start, stop, path=None, **kwargs): 
        if path is None:
            path = NOTES_PATH    

        self.load_vocab()

        if self.verbose >= 1:
            print "Constructing document-term matrix..."
        mat = self.construct_matrix(start, stop, path=path)

        if self.verbose >= 1:
            print "Fitting topic model..."
        super(TopicModel, self).fit(mat, **kwargs)


    def transform(self, start, stop, path=None, **kwargs):
        if path is None:
            path = NOTES_PATH

        if not hasattr(self, "data_file"):
            self.data_file = open(path, "r")

        mat = self.construct_matrix(start, stop, data_file=self.data_file)
        return super(TopicModel, self).transform(mat, **kwargs)


    def construct_matrix(self, start, stop=None, path=None, 
            data_file=None, remove_2014=True):
        notes = []
        term_ids = []
        if hasattr(self, "vocab"):
            vocab_size = len(self.vocab) + 1
        else:
            vocab_size = MAX_TERM

        if path is None and data_file is None:
            raise ValueError("Please provide at least a path of data_file")

        if data_file is None:
            data_file = open(path, "r")

        idx = 0
        for line in get_next_lines(data_file, start, stop):
            patient_id, note_id, year, terms = parse_line(line) 

            # Skip header
            if patient_id == "patient_id":
                continue

            for term in terms:
                try:
                    notes.append(idx)

                    if hasattr(self, "vocab"):
                        term_id = self.vocab.get(int(term) - 1, vocab_size - 1)
                    else:
                        term_id = int(term) - 1

                    term_ids.append(term_id)
                except ValueError:  # unexpected spaces in dump
                    continue

            idx += 1

        mat = coo_matrix(([1.] * len(notes), (notes, term_ids)), shape=(idx,
            vocab_size))
        return csr_matrix(mat)


    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            return cPickle.load(f)


    def save(self, path):
        with open(path, "w") as f:
            cPickle.dump(self, f)
        if self.verbose >= 1:
            print "Saved topic model."


    def load_vocab(self, path=None):
        if path is None:
            path = self.VOCAB_PATH

        vocab = []
        with open(path, "r") as f:
            for line in f:
                terms = [int(idx) for idx in line.strip().split()]
                vocab.extend(terms)

        self.vocab = dict(zip(vocab, range(len(vocab))))



if __name__ == "__main__":
    a = TopicModel(
            n_topics=300, 
            learning_method="online",
            n_jobs=1, 
            verbose=2,
            max_iter=2
    )

    a.fit(start=100, stop=None)
    a.save("../models/model.pkl")
    tr1 = a.transform(start=100, stop=None) 
    
    b = TopicModel.load("../models/model.pkl")
    tr2 = b.transform(start=100, stop=None)
    
    np.testing.assert_allclose(tr1, tr2)

