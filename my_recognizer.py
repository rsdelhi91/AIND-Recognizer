import warnings
import operator
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = [''] * test_set.num_items

    # TODO implement the recognizer
    for word_id, (features, lengths) in test_set.get_all_Xlengths().items():
      prob = {}
      for word, model in models.items():
        try:
          if model:
            prob[word] = model.score(features, lengths)
          else:
            prob[word] = float('-inf')
        except:
          prob[word] = float('-inf')
      best_guess = max(prob.items(), key=operator.itemgetter(1))
      probabilities.append(prob)
      guesses[word_id] = best_guess[0]
    return probabilities, guesses
