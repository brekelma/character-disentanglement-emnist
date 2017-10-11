import numpy as np
from emnist import EMNIST

# example replicating the dataset in the paper: 
length = 3
emnist = EMNIST()
top_words = emnist.top_words_of_length(length, max_words=300)
letters = emnist.top_letters_by_position(top_words, n = 8)
print('Length ', length, ' Words combined using letters: ')
print(letters)
train_words, test_words = emnist.valid_words_from_letters(letters)
x_train, y_train = emnist.get_data(train_words, data = 'train', per_word = 1, resample_letters = 'none', save_all = True)
x_test, y_test = emnist.get_data(test_words, data = 'test', per_word = 1, resample_letters = 'none')

# primitive example using top 300 occuring words, sampled according to occurence frequency with new letter samples for each word.
length = 6
emnist = EMNIST()
top_words, probabilities = emnist.top_words_of_length(length, max_words=300, return_probabilities = True)
samples = 500
x_train, y_train = emnist.get_data(top_words, per_word = np.ceil(probabilities*samples).astype(int), resample_letters = 'words')
print(x_train.shape) # note, we have fewer than 500 samples due to rounding to integers
