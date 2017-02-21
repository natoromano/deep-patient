# Implementation of Deep Patient

## Recap

A set of stacked denoising auto-encoders is trained on patient data (cut at
december 31, 2013), and then
evaluated by fitting random forests predicting the development of a new disease
in 2014 to the learned representation.

### Data

Each data point is a patient ID, and covariates (besides the one from topic
modeling) are medical descriptors (demographics, medical codes, drug orders). It
is first processed as follows:
- patients with fewers than 5 records before 2014 are removed
- descriptors present in less than 5 patients or in more than 80% of patients
  are removed

80,000 patients with at least 10 records before 2014, and at least one new ICD-9
code in 2014, are removed from the set of patients. They will be the *test set*
in the downstream task of predicting future disease.

The representation is learned on the remaining patients, using the
auto-encoders.

200,000 patients are randomly sampled from this set, and constitute the
*training set* for the downstream disease prediction task.

### Topic Model

Free-text medical notes are annotated using the Open Biomedical Annotator, and
terms tagged as negated or family history were removed. The paper vaguely
mentions *"analyz[ing] similarities in the representation of temporally consecutive
notes to remove duplicated information*". It is not clear how this is done in
practice.

The paper claims having about 2 millions "normalized tags". It is not clear
whether that refers to the vocabulary size used in topic modelling.

Topic modelling is done using Latent Dirichlet Allocation, and validated via
perplexity on 1 million notes (there are about 24 million notes in our dataset).
300 topics are used in the final model.

Note representations are simply averaged by-patient (before the cutting point of December 31,
2013), adding 300 covariates by patient.

### Auto-encoder

The paper uses a stack of three denoising auto-encoders, trained greedily
(layerwise). Each autoencoder uses tied weights, sigmoid activation functions,
and 500 hidden units. They use reconstruction cross-entropy loss as objective
function. The input noising is done by randomly masking 5% of the input batches.

### Evaluation

Evaluation is done by training a random forest to predict the probability of
developing each possible disease in 2014 (ICD9 are mapped to 79 medical diseases
using a Mount Sinai internal tool). The authors used 100 trees per random
forest, and trained them for each disease on 200,000 sampled patients. These
random forests are validated and tested on the 80,000 initially held back
patients.


## How to use this code

TODO
