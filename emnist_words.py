import os
import numpy as np
import string
import random
from scipy import ndimage
import cPickle
from urllib import urlretrieve
import zipfile
import gzip
import shutil
import struct 
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

path = os.getcwd()

def top_words(top_words_path = 'top_words.txt'):
    text_file = open(os.path.join(path, top_words_path))
    lines = text_file.read().split('\r')
    topwords = []
    for line in lines:
        topwords.append(line)
    return np.array(topwords[:100])

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read_data(dataset = "training",  letters_path = 'data', elements = 26):

    if dataset is "training":
        fname_img = os.path.join(path, letters_path, 'emnist-letters-train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, letters_path, 'emnist-letters-train-labels-idx1-ubyte')

    elif dataset is "testing":
        fname_img = os.path.join(path, letters_path, 'emnist-letters-test-images-idx3-ubyte')
        fname_lbl = os.path.join(path, letters_path, 'emnist-letters-test-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    try:
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)
    except:
        try:
            os.mkdir(os.path.join(path, letters_path))
        except:
            pass
        try:
            os.mkdir(os.path.join(path, letters_path, zip_path))
        except:
            pass
        fn = os.path.join(path, letters_path, 'emnist.zip')
        if not os.path.isfile(fn):
            print 'downloading data set.  this may take some time!'
            url = urlretrieve('http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip', fn)   

        zip_path = 'gzip'
        with zipfile.ZipFile(fn, 'r') as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
                if filename in ['emnist-letters-train-images-idx3-ubyte.gz', 'emnist-letters-test-images-idx3-ubyte.gz', 'emnist-letters-train-labels-idx1-ubyte.gz', 'emnist-letters-test-labels-idx1-ubyte.gz']:
                    zip_file.extract(member, os.path.join(path, letters_path))
                    f = gzip.open(os.path.join(path, letters_path, member), 'rb')
                    content = f.read()
                    f.close()
                    target = open(os.path.join(path, letters_path, os.path.splitext(filename)[0]), 'wb')
                    target.write(content)
                    target.close()
                    shutil.rmtree(os.path.join(path, letters_path, member), ignore_errors = True)
            zip_file.close()
        shutil.rmtree(os.path.join(path, letters_path, zip_path), ignore_errors = True)
        
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)
   
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
   
    get_img = lambda idx: (lbl[idx], img[idx])


    letters = {x:[] for x in range(1,elements+1)}
    for i in xrange(len(lbl)):
        letters[lbl[i]].append(img[i])
    for i in range(1,elements+1):
        letters[i] = np.array(letters[i])#.reshape((len(letters[i]), square_dim))
    return letters



def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('right')
    pyplot.show()


def get_data(per_word = 600, seed = 0, top_words = top_words(), show_img = False, word_len = 3):
    np.random.seed(seed)
    letters_dict = read_data(dataset = "training")
    original_dim = word_len*letters_dict[1][1].shape[0]*letters_dict[1][1].shape[1]
    word_imgs = np.zeros((per_word*len(top_words), original_dim))
    #word_imgs = {x:[] for x in range(len(top_words))} 
    for word_idx in range(len(top_words)):
        size = 1
        img_count = 0
        letter_idxs = []
        for lettr in top_words[word_idx]:
            letter_idx = string.lowercase.index(lettr)+1
            letter_idxs.append(letter_idx)
            size = size * letters_dict[letter_idx].shape[0]
        for m in range(per_word):
            # TODO: same letter twice in word gets same sample of handwritten letter? or make both upper/lower case? 
            i = np.random.randint(0, len(letters_dict[letter_idxs[0]])-1)
            j = np.random.randint(0, len(letters_dict[letter_idxs[1]])-1)
            k = np.random.randint(0, len(letters_dict[letter_idxs[2]])-1)
     
            temp = np.concatenate((letters_dict[letter_idxs[0]][i+1], letters_dict[letter_idxs[1]][j]), axis = 0)
            temp = np.concatenate((temp, letters_dict[letter_idxs[2]][k]), axis = 0)
            word_imgs[img_count] = np.reshape(temp.T, (original_dim))

            if show_img is True:
                show(np.reshape(word_imgs[img_count], (28,84)))
            img_count = img_count + 1
    return word_imgs

if __name__ == "__main__":
    word_imgs = get_data(per_word = 600, seed = 0, show_img = False)
    

    #ica = FastICA(n_components=26)
    #print word_imgs.shape
    #S_ = ica.fit_transform(word_imgs)  # Reconstruct signals
    #A_ = ica.mixing_ 
    #plt.plot(S_)

    #cPickle.dump(S_, open('s_.pickle', 'wb'))
    #cPickle.dump(word_imgs, open('emnist_word_data.pickle', 'wb'))
