# three-letter-mnist

Handwritten Character Disentanglement Benchmark compiled from EMNIST dataset

EMNIST class reads in handwritten character images and allows for flexible creation of word recognition datasets.  

See examples.py for usage.  Code below replicates dataset used in : Ver Steeg et. al "Disentangled Representations via Synergy Minimization" 


```
from emnist import EMNIST
import numpy as np

length = 3
emnist = EMNIST()
top_words = emnist.top_words_of_length(length, max_words=300)
letters = emnist.top_letters_by_position(top_words, n = 8)
print('Length ', length, ' Words combined using letters: ')
print(letters)
train_words, test_words = emnist.valid_words_from_letters(letters)
x_train, y_train = emnist.get_data(train_words, data = 'train', per_word = 1, resample_letters = 'none', save_all_imgs = True)
x_test, y_test = emnist.get_data(test_words, data = 'test', per_word = 1, resample_letters = 'none')
```
