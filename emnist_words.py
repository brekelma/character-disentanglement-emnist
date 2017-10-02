import os
import sys
sys.path.insert(0, '../disentanglement/utils')
import utilities
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
from pdb import set_trace as bp
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne.io import RawArray
import enchant
from sklearn.metrics import log_loss

path = os.getcwd()

def get_emnist_lettercomb(letters, sample='uniform', seed=0):
    #Returns EMNIST 3-letter-word images with given letters dictionary {}:  letters[0] = 1st letter, etc. 
    d = enchant.Dict("en_US")
    words = []
    test_words = []
    for i in letters[0]:
        for j in letters[1]:
            for k in letters[2]:
                if d.check(str(i+j+k)):
                    words.append(str(i+j+k))
                else:
                    test_words.append(str(i+j+k))
    words = np.array(words)
    print "*** WORDS *** ", words
    test_words = np.array(test_words)
    n_words = words.shape[0]
    n_words_test = test_words.shape[0]
    prob = np.array([1./n_words]*n_words)
    probt = np.array([1./n_words_test]*n_words_test)
    per_word = np.array(np.multiply(n_words,prob)).astype(int)
    per_word_test = np.array(np.multiply(n_words_test,probt)).astype(int)
    x_train, x_test, y_train, y_test = emnist_data(per_word = per_word, per_word_test = per_word_test, sample = sample, words = words, test_words = test_words, seed = 0)
    return x_train, x_test, y_train, y_test

def emnist_data(per_word = 500, per_word_test = 500, sample = 'normal', words = True,  test_words = True, original_dim = 28*84, seed = 0, set_letters = True):
    x_train, y_train, letters_ind = get_data(per_word = per_word, data = 'train', addl_path = '../three-letter-mnist', get_tw = words, sample = sample, seed = seed, set_letters = set_letters)
    x_test, y_test, a = get_data(per_word = per_word_test, data = 'train', addl_path = '../three-letter-mnist', get_tw = test_words, sample = sample, seed = seed, set_letters = letters_ind)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if x_train.shape[1] != original_dim:
        x_train = x_train.reshape((x_train.shape[0], original_dim))
    if x_test.shape[1] != original_dim:
        x_test = x_test.reshape((x_test.shape[0], original_dim))
    return x_train, x_test, y_train, y_test


def get_top_words(num_words = 100, folder = path, top_words_path = 'top_words.txt'):
    text_file = open(os.path.join(folder, top_words_path))
    lines = text_file.read().split('\r')
    topwords = []
    for line in lines:
        topwords.append(line)
    return np.array(topwords[:num_words])

def get_words_and_probs(folder = path, top_words_path = 'top_words_and_probs.txt'):
    text_file = open(os.path.join(folder, top_words_path))
    lines = text_file.read().split('\r')
    probs = []
    topwords = []
    for line in lines:
        print repr(line), repr(line[3:])
        topwords.append(str(line[:3]))
        probs.append(float(line[3:]))
    return np.array(topwords), np.array(probs) 

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read_data(dataset = "train",  letters_path = 'data', elements = 26):

    if dataset is "train":
        fname_img = os.path.join(path, letters_path, 'emnist-letters-train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, letters_path, 'emnist-letters-train-labels-idx1-ubyte')

    elif dataset is "test":
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
        print fn
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


    letters = {x:[] for x in range(elements)}
    for i in xrange(len(lbl)):
        letters[lbl[i]-1].append(img[i])
    for i in range(elements):
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
    vmax = np.max(np.abs(image))
    imgplot = ax.imshow(image, cmap=mpl.cm.seismic, vmin=-vmax, vmax=vmax)
    imgplot.set_interpolation('nearest')
    #ax.axis('off')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('right')
    pyplot.show()

def save(image, fn, save_path = 'examples'):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    vmax = np.max(np.abs(image))
    imgplot = ax.imshow(image, cmap=mpl.cm.seismic, vmin=-vmax, vmax=vmax)
    imgplot.set_interpolation('nearest')
    #ax.axis('off')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('right')
    try:
        os.mkdir(os.path.join(path, save_path))
    except:
        pass
    fig.savefig(os.path.join(path, save_path, fn))
    pyplot.close('all')

def save_all(images, fn, save_path = 'examples', method = 'ICA'):
    from matplotlib import pyplot
    import matplotlib as mpl
    #pltsize = 2*len(images)
    #print len(images)
    fig, axes = pyplot.subplots(int(np.sqrt(len(images))), int(len(images))/int(np.sqrt(len(images))), sharex = True, sharey = True)
    fig.set_figheight(4)
    fig.set_figwidth(9)
    pyplot.subplots_adjust(wspace=0, hspace=0) 
    for i in range(int(np.abs(np.sqrt(len(images))))):
        for j in range(int(len(images))/int(np.sqrt(len(images)))):
            ind = int(len(images))/int(np.sqrt(len(images)))*(i)+j
            print ind
            print images[ind].shape
            vmax = np.max(np.abs(images[ind]))
            imgplot = axes[i, j].imshow(images[ind], cmap=mpl.cm.seismic, vmin=-vmax, vmax=vmax)
            # imgplot.set_interpolation('nearest')
            axes[i,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',
                left = 'off',         # ticks along the top edge are off
                labelbottom='off',
                labelleft = 'off') # labels along the bottom edge are off


    #ax.axis('off')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('right')
    try:
        os.mkdir(os.path.join(path, save_path))
    except:
        pass
    pyplot.suptitle(str(method + " Latent Factors"), fontsize=12, fontweight = 'heavy')
    fig.tight_layout()
    pyplot.subplots_adjust(wspace=0, hspace=0) 
    plt.subplots_adjust(top=0.99)
    fig.savefig(os.path.join(path, save_path, fn))
    pyplot.close('all')

def get_seq_data(total_words = 50000, seed = 0, data = 'train', num_el = 26, addl_path = None, word_len = 3, save_img = False):
    if addl_path is not None:
        path = addl_path
    else:
        path = os.getcwd() 

    per_word = int(total_words/int(num_el/3.))
    np.random.seed(seed)
    letters_dict = read_data(dataset = data)
    original_dim = word_len*letters_dict[1][1].shape[0]*letters_dict[1][1].shape[1]
    word_imgs = np.zeros((per_word*int(num_el/3), original_dim))
    word_labels = np.chararray((per_word*int(num_el/3)))
    #word_imgs = {x:[] for x n range(len(top_words))} 
    img_count = 0
    letter = 1
    while letter <= num_el - word_len: 
        size = 1
        letter_idxs = []
      
        #for lettr in top_words[word_idx]:
        #    letter_idx = string.lowercase.index(lettr)+1
        #    letter_idxs.append(letter_idx)
        #    size = size * letters_dict[letter_idx].shape[0]
        for m in range(per_word):
            #print letter
            # TODO: same letter twice in word gets same sample of handwritten letter? or make both upper/lower case? 
            i = np.random.randint(0, len(letters_dict[letter])-1)
            j = np.random.randint(0, len(letters_dict[letter+1])-1)
            k = np.random.randint(0, len(letters_dict[letter+2])-1)

            temp = np.concatenate((letters_dict[letter][i], letters_dict[letter+1][j]), axis = 0)
            temp = np.concatenate((temp, letters_dict[letter+2][k]), axis = 0)
            word_imgs[img_count] = np.reshape(temp.T, (original_dim))
            word_labels[img_count] = str(chr(ord('a')+letter-1) + ' '+ chr(ord('a')+letter) + ' ' + chr(ord('a')+letter+1))
            #if m % 1000 == 0:
            #    print word_labels[img_count]

            if save_img is True:
                fn = word_labels[img_count]+'_'+ str(m)+ '.pdf'
                #show(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0]))
                save(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0])), fn)

            img_count = img_count + 1
        letter = letter + 3
    return word_imgs, word_labels


def get_data(per_word = 500, seed = 0, data = 'train', num_words = 100, save_img = False, word_len = 3, addl_path = None, get_tw = True, sample = 'normal', set_letters = True):
    if addl_path is not None:
        path = addl_path
    else:
        path = os.getcwd() 
    if type(get_tw) is np.ndarray:
        print 'got array ', get_tw.shape
        top_words = get_tw
    elif get_tw:
        top_words = get_top_words(num_words = num_words, folder = path)
    else:
        print 'please enter numpy array or set get_top_words = True'
    np.random.seed(seed)
    letters_dict = read_data(dataset = data)
    original_dim = word_len*letters_dict[1][1].shape[0]*letters_dict[1][1].shape[1]
    if type(per_word) is np.ndarray:
        word_imgs = np.zeros((np.sum(per_word), original_dim))
        word_labels = [] #np.chararray((np.sum(per_word)))
    else:
        word_imgs = np.zeros((per_word*len(top_words), original_dim))
        word_labels = []#np.chararray((per_word*len(top_words)))
        #word_imgs = {x:[] for x n range(len(top_words))} 

    img_count = 0

    if type(set_letters) is np.ndarray:# and set_letters.shape[0] == 26:
        print 'using array for letters indices'
        letters_inds = set_letters
        set_letters = False
    else:
        letters_inds = np.zeros((26))

    for word_idx in range(len(top_words)):
        size = 1
        letter_idxs = []
        for lettr in top_words[word_idx]:
            letter_idx = ord(lettr.lower())-97
            letter_idxs.append(letter_idx)
            size = size * letters_dict[letter_idx].shape[0]
      

        if type(per_word) is np.ndarray:
            try:
                per_word_ = per_word[word_idx]
            except:
                print 'per_word array must have same length as top_words'
                return False
        else:
            per_word_ = per_word

        
        #        i = np.random.randint(0, len(letters_dict[letter_idxs[0]])-1)
        #        j = np.random.randint(0, len(letters_dict[letter_idxs[1]])-1)
        #        k = np.random.randint(0, len(letters_dict[letter_idxs[2]])-1)

        for m in range(per_word_):
        # TODO: same letter twice in word gets same sample of handwritten letter? or make both upper/lower case? 
            if set_letters:
                if m == 0 or not (sample == 'uniform'): #sample = 'normal' gives different letters within same word
                    i = np.random.randint(0, len(letters_dict[letter_idxs[0]])-1)
                    j = np.random.randint(0, len(letters_dict[letter_idxs[1]])-1)
                    k = np.random.randint(0, len(letters_dict[letter_idxs[2]])-1)

                if letters_inds[letter_idxs[0]] == 0:
                    letters_inds[letter_idxs[0]] = i
                if letters_inds[letter_idxs[1]] == 0: 
                    letters_inds[letter_idxs[1]] = j
                if letters_inds[letter_idxs[2]] == 0:
                    letters_inds[letter_idxs[2]] = k

            i = int(letters_inds[letter_idxs[0]])
            j = int(letters_inds[letter_idxs[1]]) 
            k = int(letters_inds[letter_idxs[2]])

            a = letters_dict[letter_idxs[0]]
            b = letters_dict[letter_idxs[1]]
            c = letters_dict[letter_idxs[2]]
            temp = np.concatenate((a[i],b[j]), axis = 0)
            temp = np.concatenate((temp, c[k]), axis = 0)

            word_imgs[img_count] = np.reshape(temp.T, (original_dim))
            word_labels.append(str(top_words[word_idx]))

            if save_img is True:
                fn = top_words[word_idx]+'_'+ str(m)+ '.pdf'
                #show(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0]))
                save(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0])), fn)

            img_count = img_count + 1
    #for word_idx in range(len(top_words)):
    print 'letters_inds: ', set_letters, ' : ', letters_inds
    return word_imgs, word_labels, letters_inds


def run_ica(data, n_components = None, max_iter = 200):
    ica = FastICA(n_components = n_components, max_iter = max_iter)
    return ica.fit(data)

def run_infomax(data, n_components = None, max_iter = 200):
    info = mne.create_info(data.shape[1], 1, 'eeg') #channels, sampling rate
    print 'data: ', data.shape
    raw = mne.io.RawArray(data.T, info) #transpose since 
    infomax = ICA(n_components = n_components, method = 'extended-infomax', max_iter = max_iter)
    picks = mne.pick_types(raw.info, eeg = True, exclude='bads')
    infomax.fit(raw, picks=picks)
    #A_infomax = infomax.mixing_matrix_
    return infomax, raw

def get_top_letter_words(letters = None):
    d = enchant.Dict("en_US")
    words = []
    test_words = []
    #letters = np.array(['e','o','a','t','n','y','s','r','u','d','i','l','w','h','b','g'])
    if letters == None:
        letters = {}
        letters[0] = ['a','o','b','l','h','s','t','c']#,'e','w','d','g','p','f','m','n']
        letters[1] = ['e','a','o','i','u','n','s','r']#,'l','w','g','f','t','c','d','y']
        letters[2] = ['t','e','y','n','r','d','w','o']#,'x','s','l','g','f','p','b','m']
    #old (by freq) ['t', 'a', 's', 'h','f','o','y','n','b','w','c','g','l','m','d', 'p']
    #old (by freq) ['h', 'n','o','a', 'e','u','i','l','s','w','r', 't','f','g','y', 'd']
    #old (by freq) ['e','d','t','r','y','u']#,'s','n','w','o','l','m','g','p','x', 'k']
    for i in letters[0]:
        for j in letters[1]:
            for k in letters[2]:
                if d.check(str(i+j+k)):
                    words.append(str(i+j+k))
                    print str(i+j+k)
                else:
                    #words.append(str(i+j+k))
                    test_words.append(str(i+j+k))
    return np.array(words), np.array(test_words)

def calc_score(data, square = False):
    if not (data.shape[0] == 28 and data.shape[1] == 84):
        print 'please give 28x84 vector'
    overall_sum = 0
    char_sum = []
    for i in range(3):
        if square:
            char_sum.append(np.sum(data[:, 28*i:28*(i+1)]**2, axis= (0,1)))
        else:
            char_sum.append(np.sum(np.abs(data[:, 28*i:28*(i+1)]), axis= (0,1)))
        overall_sum = overall_sum + char_sum[i]  
    return np.divide(np.array(char_sum), overall_sum)

if __name__ == "__main__":
    #pw = 500
    tw = 8
    unique_seq = 8
    dataset = 'emnist'
    data_type = 'normal'
    test_recons = 30

    letters = {}
    letters[0] = ['s','a','b','p','l','d','f','t']
    letters[1] = ['a','e','o','i','u','s','r','n']
    letters[2] = ['t','n','y','s','d','e','m','r']
    #letters[0] = list(string.ascii_lowercase)
    #letters[1] = list(string.ascii_lowercase)
    #letters[2] = list(string.ascii_lowercase)
    
    if dataset == 'mnist':
        x_train, x_test, y_train, y_test = utilities.mnist_data()
        word_imgs = np.array(x_train[:samples, :])
        per_word = np.array([1]*10)
    else:
        if letters is not None:
                dist = 'top_'+ str(len(letters[0]))+'_letters'
                word_imgs, test_imgs, word_labels, test_labels = get_emnist_lettercomb(letters)
        else:
            sample = 'uniform'
            words, test_words = get_top_letter_words()
            #words = np.array(['pit', 'pot', 'pig', 'jot', 'jig', 'jog'])
            #prob = np.array([.125, .125, .125, .125, .125, .125, .125, .125])
            #prob = np.array([.02, .05, .18, .01, .24, .15, .05, .3])
            #words = np.array(['pit', 'pot', 'pig', 'jot', 'jig', 'jog'])#, 'wit', 'wig', 'cot', 'cog'])
            #words, prob = get_words_and_probs(top_words_path = '../three-letter-mnist/top_words_and_probs.txt')
            #n_words = 100
            #words = words[2:]
            #prob = prob[2:]/np.sum(prob[2:])
            n_words = words.shape[0]
            samples = n_words
            n_test_words = test_words.shape[0]
            n_samples = n_test_words
            #prob = np.array([.02, .05, .18, .01, .24, .15, .05, .3])
            prob = np.array([1./n_words]*n_words)
            probT = np.array([1./n_test_words]*n_test_words)
            #prob = np.array([.1, .1, .08, .08, .06, .06, .05, .05, .05, .05, .04, .04,.04, .04, .04, .04, .02, .02, .02, .02])
            print prob.shape, np.sum(prob), n_words
            per_word = np.array(np.multiply(samples,prob)).astype(int)
            per_word_test = np.array(np.multiply(n_samples,probT)).astype(int)
            #words = get_top_words(n_words)

            if np.all(prob == prob[0]):
               dist = 'uniform'
            else:
               dist = 'nonunif'
            prob = np.array([.02, .05, .18, .01, .24, .15, .05, .3])
            
            try:
               num_words = words.shape[0]
            except:
               num_words = 100

            if data_type == 'seq':
               word_imgs, word_labels = get_seq_data(total_words = tw, save_img = True)
            else:
               word_imgs, word_labels, letters_inds = get_data(per_word = per_word, seed = 0, num_words = n_words, save_img = False, sample = sample, get_tw = words)
               test_imgs, test_labels, a = get_data(per_word = per_word_test, seed = 0, num_words = n_test_words, save_img = False, sample = sample, get_tw = test_words, set_letters = letters_inds)
               #word_imgs, word_labels = get_data(per_word = pw, seed = 0, num_words = 100, save_img = True, sample = 'uniform')
        print 'WORD IMGs SHAPE: ', word_imgs.shape

    # word_imgs = np.concatenate((word_imgs, np.concatenate((word_imgs, word_imgs), axis = 0)), axis = 0)
    # test_imgs = np.concatenate((test_imgs, np.concatenate((test_imgs, test_imgs), axis = 0)), axis = 0)
    # word_labels = np.concatenate((word_labels, np.concatenate((word_labels, word_labels), axis = 0)), axis = 0)
    # test_labels = np.concatenate((test_labels, np.concatenate((test_labels, test_labels), axis = 0)), axis = 0)

    for n in [9]:
        print 'running ICA with ' + str(n) + ' components'
        ica = run_ica(word_imgs, n_components = n)
        infomax, raw = run_infomax(word_imgs, n_components = n)
        #change to directly handle ica, infomax
        for method in ['ICA', 'InfoMax']:
            if method == 'ICA':
                A_ = ica.mixing_
                W_ = ica.components_ #unmixing matrix from x to s 
            else:
                A_ = infomax.get_components()
                W_ = infomax.unmixing_matrix_.T
                info = mne.create_info(test_imgs.shape[1], 1, 'eeg')
                raw_t = mne.io.RawArray(test_imgs.T, info)
                infomax.apply(raw_t, n_pca_components = infomax.pca_components_.shape[0])
                data, times = raw_t[:,:]
                reconstructions = data.T
                # data, times = infomax.get_sources(raw)[:,:]
            #A_  = ica.mixing_matrix_
            if dataset == 'mnist':
                top_path = method + '_' + dataset
            else:
                top_path =  method + '_' + dataset + '_' +  dist
            try:
                os.mkdir(os.path.join(path, top_path))
            except:
                pass
            components = {}
            scores = np.zeros((A_.shape[1],3))
            
            #save components
            for i in range(A_.shape[1]):
                if method == 'ICA':
                    if dataset =='mnist':
                        component = np.reshape(A_[:,i],(28,28))
                    else:
                        component = np.reshape(A_[:,i],(28,84))
                    components[i] = component
                    scores[i] = calc_score(np.reshape(A_[:,i],(28,84)))
                    method_path = os.path.join(top_path, 'ica components n='+str(n))
                    try:
                        os.mkdir(os.path.join(path, method_path))
                    except:
                        pass
                      
                else:
                    if dataset =='mnist':
                        component = np.reshape(A_[:,i],(28, 28))
                    else:
                        #what to do here?  errors?
                        component = A_[:,i].reshape((28,84))
                    components[i] = component
                    scores[i] = calc_score(np.reshape(A_[:,i],(28,84)))
                    method_path = os.path.join(top_path, 'infomax components n='+str(n))
                    try:
                       os.mkdir(os.path.join(path, method_path))
                    except:
                       pass
                save(component, 'component_'+str(i)+'_'+str(n)+'.pdf', save_path = method_path)  
            save_all(components, 'components_'+str(n)+'.pdf', save_path = method_path, method = method)

            print 'CLUSTER SCORES'
            print scores


            indices = np.random.randint(0, test_imgs.shape[0]-1, test_recons)
 
            for i in indices:
                if method == 'ICA':
                    recon_img = ica.inverse_transform(ica.transform(test_imgs[i,:].reshape(1, -1))).reshape((28,84))
                    save(recon_img, 'test_'+str(test_labels[i])+'.pdf', save_path = method_path) 
                else:
                    recon_img = reconstructions[i,:].reshape((28,84))
                    save(recon_img, 'test_'+str(test_labels[i])+'.pdf', save_path = method_path)
                    
            # for i in range(per_word_array.shape[0]):  
            #     idx = np.sum(per_word_array[:i])
            #     #print method, ': ', idx
            #     if method == 'ICA':
            #         #i counts in perword array (+1, cumulative sum)
            #         if dataset == 'mnist':
            #             recon_img = ica.inverse_transform(ica.transform(word_imgs[idx,:].reshape(1, -1))).reshape((28,28))
            #         else:
            #             recon_img = ica.inverse_transform(ica.transform(word_imgs[idx,:].reshape(1, -1))).reshape((28,84))
            #         save(recon_img, 'recon_'+str(idx)+'_'+str(n)+'.pdf', save_path = method_path)
                #else:
                    # if dataset == 'mnist':
                    #     recon_img = reconstructions[idx,:].reshape((28,28))
                    # else:
                    #     recon_img = reconstructions[idx,:].reshape((28,84))
                    # save(recon_img, 'recon_'+str(idx)+'_'+str(n)+'.svg', save_path = method_path)

        # if data_type == 'seq':
        #     inc = int(tw/unique_seq)
        #     for j in range(unique_seq):
        #         recon = np.reshape(ica.inverse_transform(ica.transform(np.reshape(word_imgs[j*inc+1, :],(1,-1)))), (28,84))
        #         true = np.reshape(word_imgs[j*inc+1, :], (28,84))
        #         ica_path = os.path.join(top_path, 'recon n='+str(n))
        #         try:
        #             os.mkdir(os.path.join(path, top_path))
        #         except:
        #             pass
        #         save(np.concatenate((recon, true), axis=1), 'recon_'+str(j)+'_'+str(n)+'.svg', save_path = ica_path)
        # else:
        #     for j in range(100):
        #         recon = np.reshape(ica.inverse_transform(ica.transform(np.reshape(word_imgs[j*pw, :],(1,-1)))), (28,84))
        #         true = np.reshape(word_imgs[j*pw, :], (28,84))
        #         ica_path = os.path.join(top_path, 'recon n='+str(n))
        #         try:
        #             os.mkdir(os.path.join(path, top_path))
        #         except:
        #             pass
        #         save(np.concatenate((recon, true), axis=1), 'recon_'+str(j)+'_'+str(n)+'.svg', save_path = ica_path)

    #plt.plot(S_)

    #cPickle.dump(S_, open('s_.pickle', 'wb'))
    #cPickle.dump(word_imgs, open('emnist_word_data.pickle', 'wb'))
