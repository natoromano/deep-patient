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
import argparse
import collections

import sklearn
from sklearn.decomposition import LatentDirichletAllocation

NOTES_PATH = '/scratch/users/kjung/ehr-repr-learn/data/notes_cui.csv'
MAX_TERM = 7299366
NUM_NOTES = 28000000



# Utility functions

def get_next_lines(data_file, start, stop=None):
    """
    Returns a generator with lines in file.

    Note: indices must be slide.
    """
    if stop is None:
        return (line for line in islice(data_file, start))
    else:
        return (line for line in islice(data_file, start, stop))


def parse_line(line):
    """
    Returns:
        patient_id, note_id, note_year, list of terms
    """
    split = line.strip().split("\t")
    if len(split) == 5:
        patient_id, note_id, age, note_year, terms = split
    else:
        patient_id, note_id, age, note_year = split
        terms = ""
    return patient_id, note_id, note_year, terms.split(" ")


def encode_notes(n_topics, n_jobs, start_idx, end_idx, model_path,
        notes_path, embedding_path, vocab_path, batch_size, use_saved=False, verbose=2):
    """
    Trains a topic model on the provided notes, and encodes all the others.

    start_idx and end_idx provide the scope of the training set.
    if end_idx is None, start_idx become the end index and the start index is 0.
    """
    if not use_saved:
        model = TopicModel(
                n_topics=n_topics, 
                learning_method="online",
                n_jobs=n_jobs, 
                verbose=verbose,
        )

        model.fit(start=start_idx, stop=end_idx, vocab=vocab_path)
        model.save(model_path)
    else:
        model = TopicModel.load(model_path)

    embeddings = collections.defaultdict(dict)  # patient_id: note_id: encoding

    print "Encoding data..."

    num_batches = int(float(NUM_NOTES) / batch_size)
    for i in xrange(num_batches):
        try:
            encoded, ids = model.transform(start=batch_size, stop=None)
        except ValueError:
            print "Reached end of data."
            break  # End of data
        for j, (pat_id, note_id) in enumerate(ids):
            embeddings[int(pat_id)][int(note_id)] = encoded[j, :]
        
        if verbose >= 2:
            print "Encoded batch %d with shape" % i, encoded.shape

    # Construct set of input note IDs
    note_ids = set()
    with open("data/input.notes.txt") as f:
        for line in f:
            try:
                note_ids.add(int(line.strip()))
            except ValueError:
                continue

    # Start dumping notes
    per_patient = embedding_path

    # Create empty files
    open(per_patient, "a").close()

    # We only embed per patients here
    with open(per_patient, "w") as g:
        for patient_id, notes in embeddings.iteritems():
            cPickle.dump((patient_id, np.mean([n for nid, n in notes.iteritems() if
                nid in note_ids], axis=0)), g)

    print "Done!"



# Topic Modeling

class TopicModel(LatentDirichletAllocation):
    """
    Wrapper around sklearn's LDA model to handle dumped data.
    """

    def fit(self, start, stop, vocab, path=None, **kwargs): 
        """
        Wrapper around fitting methods, to first construct the matrix 
        from indices between start and stop.
        """
        if path is None:
            path = NOTES_PATH    

        self.load_vocab(vocab)

        if self.verbose >= 1:
            print "Constructing document-term matrix..."
        mat = self.construct_matrix(start, stop, path=path)

        if self.verbose >= 1:
            print "Fitting topic model..."
        super(TopicModel, self).fit(mat, **kwargs)
        del mat


    def transform(self, start, stop, path=None, **kwargs):
        """
        Wrapper around transform methods, to first construct the matrix 
        from indices between start and stop.
        """
        if path is None:
            path = NOTES_PATH

        if not hasattr(self, "data_file"):
            self.data_file = open(path, "r")

        mat, ids = self.construct_matrix(start, stop, data_file=self.data_file,
                return_ids=True)
        return super(TopicModel, self).transform(mat, **kwargs), ids


    def construct_matrix(self, start, stop=None, path=None, 
            data_file=None, return_ids=False):
        """
        Constructs the document-term matrix.
        
        If only start is given, will take notes until @start idx. If both start
        and end are given, will take notes between @start and @end.
        """
        notes = []
        term_ids = []
        ids = []

        if hasattr(self, "vocab"):
            vocab_size = len(self.vocab) + 1
        else:
            vocab_size = MAX_TERM

        if path is None and data_file is None:
            raise ValueError("Please provide at least a path of data_file")

        if data_file is None:
            data_file = open(path, "r")


        if start is None and stop is None: 
            lines_generator = data_file 
        else:
            line_generator = get_next_lines(data_file, start, stop)

        idx = 0
        for line in get_next_lines(data_file, start, stop):
            patient_id, note_id, year, terms = parse_line(line) 
            # Skip header
            if patient_id == "patient_id" or len(terms) == 0:
                continue

            # Save ids for encoding
            ids.append((patient_id, note_id))

            for term in terms:
                try:

                    if hasattr(self, "vocab"):
                        term_id = self.vocab.get(int(term) - 1, vocab_size - 1)
                    else:
                        term_id = int(term) - 1
                    
                    notes.append(idx)
                    term_ids.append(term_id)
                except ValueError:  # unexpected spaces in dump
                    continue

            idx += 1

        mat = coo_matrix(([1.] * len(notes), (notes, term_ids)), shape=(idx,
            vocab_size))
        if return_ids:
            return csr_matrix(mat), ids
        else:
            return csr_matrix(mat)


    @classmethod
    def load(cls, path):
        """
        Loads saved model.
        """
        with open(path, "r") as f:
            return cPickle.load(f)


    def save(self, path):
        """
        Saves model to given file.
        
        Will overwrite it.
        """
        with open(path, "w") as f:
            cPickle.dump(self, f)
        if self.verbose >= 1:
            print "Saved topic model."


    def load_vocab(self, path):
        """
        Loads the vocabulary (list of words/cuids to use). All other words will
        be marked as "out of vocab".
        """
        vocab = []
        with open(path, "r") as f:
            for line in f:
                term = int(line.strip())
                vocab.append(term)

        self.vocab = dict(zip(vocab, range(len(vocab))))



if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Parameters')
   parser.add_argument('-t', '--topics', type=int, default=300)
   parser.add_argument('-j', '--jobs', type=int, default=1)
   parser.add_argument('-s', '--start', type=int, default=100000)
   parser.add_argument('-e', '--end', type=int, default=None)
   parser.add_argument('-m', '--model', type=str, default="models/model_cui.pkl")
   parser.add_argument('-d', '--notes', type=str, default=NOTES_PATH)
   parser.add_argument('-g', '--embedding', type=str,
    default="/scratch/users/naromano/deep-patient/shah/embedding_patients_cui_pkl")
   parser.add_argument('--vocab', default="data/cuis_thr_50.txt")
   parser.add_argument('-v', '--verbose', type=int, default=0)
   parser.add_argument('-b', '--batch', type=int, default=100000)
   parser.add_argument('-l', '--load', dest='load', action="store_true")
   parser.set_defaults(load=False)
   args = parser.parse_args()

   encode_notes(args.topics, args.jobs, args.start, args.end, args.model,
           args.notes, args.embedding, args.vocab, args.batch, args.load, args.verbose)

