import numpy as np
import os
from scipy import ndimage
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil
import struct
import enchant
from itertools import product

class EMNIST():
    def __init__(self, data_path = None, save_path = None):

        self.cwd = os.getcwd()
        self.save_path = save_path if save_path is not None else os.path.join(self.cwd, 'examples')
        self.data_path = data_path if data_path is not None else os.path.join(self.cwd, 'data')
        
        # data = dictionary with keys as letter index 1-26, values: 4800 images samples of 28x28 size
        self.train_data = self.read_data(dataset = 'train')
        self.test_data = self.read_data(dataset = 'test')

        
    def top_words_of_length(self, length, max_words = 1000, file_path = 'all_words.txt', return_probabilities = False):
        '''	Base method for finding top words of given length in txt file lists of words.  File all_words.txt populate top 300k + words with occurence counts 
            (words & frequencies from  http://norvig.com/ngrams/count_1w.txt)

            max_words = int or None:  Max number of words to return, or None to not limit maximum number
            length = word length to search for
            file_path : to all_words.txt or similar file with tab delimited .txt: "word \t count \n"
            return_probabilities: calculate and return relative word-occurence statistics
        '''
        top_words = []
        probabilities = []
        text_file = open(file_path)
        lines = [line.split('\t') for line in text_file.read().splitlines()]
        for line_list in lines:
            if max_words is None or len(top_words) < max_words:
                if len(line_list[0]) == length:
                    top_words.append(str(line_list[0]))
                    if return_probabilities:
                        probabilities.append(float(line_list[1]))
            if max_words is not None and len(top_words) == max_words:
                break

        if return_probabilities:
            total = sum(probabilities)
            probabilities = np.array([i / total for i in probabilities])
            return top_words, probabilities
        else:
            return top_words



    def valid_words_from_letters(self, letters, return_test = True):
        '''Separates combinations of given letters into lists of well defined English words and not.  
            Valid words can be used as training set.  
            return_test is a bool for whether to give invalid words as test set, True by default
            
            Input: letters = List of lists of letters in each position (Larger list has n_elements = length of word)
                e.g. letters[1] = ['a', 'e', 'i', 'o', 'u']
        '''
        d = enchant.Dict("en_US")
        words = []
        test_words = []
        for word in product(*letters):
            word_ = ''.join( ltr for ltr in word )
            if d.check(word_):
                words.append(word_)
            else:
                test_words.append(word_)
        return words, test_words if return_test else words


    def top_letters_by_position(self, words, n = 8):
        ''' Get the n top occuring letters in each position for the given words. Assumes all words are of the same length'''
        try:
            print(len(words), len(words[0]), len(words[-1]))
            positions = np.concatenate([np.array([ord(words[w][i]) for i in range(len(words[w]))])[np.newaxis, :] for w in range(len(words))], axis =0)
        except:
            raise ValueError("Please ensure all words are the same length, or that there are no extra entries in word list.")

        top_letters = ['']*positions.shape[1]
        for i in range(positions.shape[1]):
            ordered_letters = np.flip(np.argsort(np.bincount(positions[:, i])), axis =0)
            top_letters[i] = [chr(ltr) for ltr in ordered_letters[:n]]

        return top_letters


    def get_data(self, words, data = 'train', resample_letters = 'none', fixed_letters = True,  
                    per_word = 1, seed = 0, save_all_imgs = False):

        ''' Converts list of words to dataset of handwritten images:
            Parameters : 
                words : list of words to form (Note: call for train and test separately)

                data : 'train' or 'test'

                resample_letters: 'none', 'all', 'words'
                    'none' : use a single image for each appearance of a letter in a word / position
                    'all' : resample image for each appearance of a letter in a word / position
                    'words' : use same image for a letter appearing multiple times in a word, but resample across words

                fixed_letters : bool, used with resample_letters = 'none' ONLY
                    If True, self.letter_samples will be used for images (useful for matching train/test data)
                    If False, letters will be resampled according to resample_letters

                per_word : list OR integer
                    integer => fixed number of image samples for each word
                    list / 1-d array with length matching # of words => # of sampled images for each word
                seed : np.random seed for sampling which image to use for each letter instance
        '''	    
        np.random.seed(seed)

        letters_dict = self.train_data if data == 'train' or data == 'training' else self.test_data

        # assumes all words / input are same dimension
        word_len = len(words[0])
        original_dim = word_len*np.prod(letters_dict[1].shape[1:])

        if type(per_word) is np.ndarray:
            word_imgs = np.zeros((np.sum(per_word), original_dim))
        elif type(per_word) is list:
            word_imgs = np.zeros(sum(per_word), original_dim)
        else:
            word_imgs = np.zeros((per_word*len(words), original_dim))
        
        word_labels = []
        img_count = 0
        letter_samples = np.zeros((26), dtype = int)
        img_sample = [0]*word_len
        
        for word_idx in range(len(words)):
            # 1-26 index of letters
            letter_idxs = []
            for lettr in words[word_idx]:
                letter_idx = ord(lettr.lower())-97
                letter_idxs.append(letter_idx)


            if type(per_word) is np.ndarray or type(per_word) is list:
                try:
                    per_word_ = per_word[word_idx]
                except:
                    raise ValueError('Length of per_word list must be equal to length of words list')
            else:
                per_word_ = per_word

            
            
            for m in range(per_word_):
                #if not fixed_letters:

                #if (not resample_letters == 'none') or (m == 0): #sample = 'normal' gives different letters within same word
                
                for pos in range(word_len):
                    img_sample[pos] = np.random.randint(0, len(letters_dict[letter_idxs[pos]])-1) 
                    if not resample_letters == 'none' or letter_samples[letter_idxs[pos]] == 0:
                        # don't resample if 'words' and already seen letter in this word
                        if resample_letters == 'words' and (pos == 0 or not letter_idxs[pos] in letter_idxs[:pos]):
                            letter_samples[letter_idxs[pos]] = int(img_sample[pos])
                            
                    # pull appropriate images from letters_dict = train/test images
                    if pos == 0:
                        np_img = np.array(letters_dict[letter_idxs[pos]][letter_samples[letter_idxs[pos]]])
                    else:
                        new_img = np.array(letters_dict[letter_idxs[pos]][letter_samples[letter_idxs[pos]]])
                        np_img = np.concatenate([np_img, new_img], axis = 0)
            
                # Rescale to be in range [0,1]
                word_imgs[img_count] = np_img.T.reshape((original_dim))/255.0
                
                if save_all_imgs:
                    fn = words[word_idx]+'_'+ str(m)+ '.pdf'
                    self.save((word_imgs[img_count]).astype(float), fn)
                
                
                word_labels.append(str(words[word_idx]))
                img_count = img_count + 1
            
        return word_imgs, word_labels


    def read_data(self, dataset = 'train', data_path = None, elements = 26):
        ''' Loads the training or test handwritten letter images into memory.  
            if necessary, downloads the dataset from http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'

            dataset : 'train or 'test' 
            data_path : path where data will be downloaded or location where it can be found (default is to use constructor value)
            elements : 26 letters
        '''

        if dataset is "train":
            fname_img = os.path.join(self.data_path, 'emnist-letters-train-images-idx3-ubyte')
            fname_lbl = os.path.join(self.data_path, 'emnist-letters-train-labels-idx1-ubyte')

        elif dataset is "test":
            fname_img = os.path.join(self.data_path, 'emnist-letters-test-images-idx3-ubyte')
            fname_lbl = os.path.join(self.data_path, 'emnist-letters-test-labels-idx1-ubyte')
        else:
            raise ValueError("dataset must be 'train' or 'test'")

        # Load everything in some numpy arrays
        try:
            with open(fname_lbl, 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)
        except:
            try:
                os.mkdir(os.path.join(self.data_path))
            except:
                pass
            #try:
            #    os.mkdir(os.path.join(self.data_path, zip_path))
            #except:
            #    pass
            fn = os.path.join(self.data_path, 'emnist.zip')
            if not os.path.isfile(fn):
                print('Downloading data set...  this may take some time!')
                url = urlretrieve('http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip', fn)   
            print(fn)
            zip_path = 'gzip'
            with zipfile.ZipFile(fn, 'r') as zip_file:
                for member in zip_file.namelist():
                    filename = os.path.basename(member)
                    # skip directories
                    if not filename:
                        continue
                    if filename in ['emnist-letters-train-images-idx3-ubyte.gz', 'emnist-letters-test-images-idx3-ubyte.gz', 'emnist-letters-train-labels-idx1-ubyte.gz', 'emnist-letters-test-labels-idx1-ubyte.gz']:
                        zip_file.extract(member, self.data_path)
                        f = gzip.open(os.path.join(self.data_path, member), 'rb')
                        content = f.read()
                        f.close()
                        target = open(os.path.join(self.data_path, os.path.splitext(filename)[0]), 'wb')
                        target.write(content)
                        target.close()
                        shutil.rmtree(os.path.join(self.data_path, member), ignore_errors = True)
                zip_file.close()
            shutil.rmtree(os.path.join(self.data_path, zip_path), ignore_errors = True)

            with open(fname_lbl, 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        get_img = lambda idx: (lbl[idx], img[idx])


        letters = {x:[] for x in range(elements)}
        for i in range(len(lbl)):
            letters[lbl[i]-1].append(img[i])
        for i in range(elements):
            letters[i] = np.array(letters[i])
        return letters

    def show(image):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        image = image.reshape((28,-1))
        imgplot = ax.imshow(image, cmap=pyplot.cm.seismic, vmin =0, vmax = 1)
        imgplot.set_interpolation('nearest')
        pyplot.show()

    def save(self, image, fn = 'example_img.pdf', save_path = 'examples'):
        from matplotlib import pyplot
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        image = image.reshape((28,-1))
        imgplot = ax.imshow(image, cmap=pyplot.cm.seismic, vmin =0, vmax = 1)
        imgplot.set_interpolation('nearest')

        try:
            os.mkdir(save_path)
        except:
            pass
        fig.savefig(os.path.join(save_path, fn))
        pyplot.close('all')
