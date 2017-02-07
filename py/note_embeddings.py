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



def construct_matrix(start, stop=None, path=None, data_file=None, remove_2014=True):
    notes = []
    term_ids = []

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
                term_ids.append(int(term) - 1)
            except ValueError:  # unexpected spaces in dump
                continue

        idx += 1

    mat = coo_matrix(([1.] * len(notes), (notes, term_ids)), shape=(idx,
        MAX_TERM))
    return csr_matrix(mat)



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



class TopicModel(LatentDirichletAllocation):
    """
    Wrapper around sklearn's LDA model to handle dumped data.
    """


    def fit(self, start, stop, path=None, **kwargs): 
        if path is None:
            path = NOTES_PATH    
        
        print "Contructing matrix... ",
        mat = construct_matrix(start, stop, path=path)
        print "Done."

        print "Fitting topic model... ",
        super(TopicModel, self).fit(mat, **kwargs)
        print "Done."


    def transform(self, start, stop, path=None, **kwargs):
        if path is None:
            path = NOTES_PATH

        if not hasattr(self, "data_file"):
            self.data_file = open(path, "r")

        mat = construct_matrix(start, stop, data_file=self.data_file)
        return super(TopicModel, self).transform(mat, **kwargs)


    @classmethod
    def load(cls, path):
        return cPickle.load(path)


    def save(self, path):
        with open(path, "w") as f:
            cPickle.dump(self, f)
        print "Saved topic model."


if __name__ == "__main__":
    print "Creating model... ",
    a = TopicModel(
            n_topics=300, 
            learning_method="online",
            n_jobs=1, 
            verbose=2,
            max_iter=2
    )
    print "Done."

    a.fit(start=100, stop=None)
    a.save("data/model.pkl")

