# character-disentanglement-emnist

Handwritten Character Disentanglement Benchmark compiled from EMNIST dataset

EMNIST class reads in handwritten character images and allows for flexible creation of word recognition datasets from the EMNIST letters found here :
https://www.nist.gov/itl/iad/image-group/emnist-dataset

See examples.py for usage.  Code below replicates dataset used in : 
Ver Steeg et. al "Disentangled Representations via Synergy Minimization" 

all_words.txt contains words listed by occurence frequency from Peter Norvig, according to Google Web Trillion Word Corpus.
http://norvig.com/ngrams/   (count_1w.txt)



```python
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

### Useful Methods & Parameters
* top_words_of_length ( length , max_words , file_path , return_probabilities)
  
  Choose word length and number of words, returns list of word strings and, optionally, relative occurence probabilities normalized amongst chosen words.  These may be fed to "get_data" directly to pull images, or used to choose commonly occuring letters in each character position.
  
  
* top_letters_by_position ( words, n )

  Choose top n letters occuring in each of the i positions in words list.  Returns list (length = word length) of lists of letters (length = n)
  
* valid_words_from_letters ( letters )

  Given list of lists of letters to go in each position, find split all combinations of letters into validly defined words (train), and not (test).  Can be used with top_letters or custom letter choices by position.
  
* get_data (words, data = 'train'/'test', per_word , resample_letters)

Construct dataset of images by specifying word list, whether to take images from EMNIST training or test data, and how many samples per_word (can be an integer or 1d array with indices matching word list).

...* resample_letters = 'none': Same letter image for each instance of a letter in a word
...* resample_letters = 'all' : Resample image for each instance of a letter in word
...* resample_letters = 'words' : Use same letter image within same word sample : i.e same 'p' within word 'pip'
   
