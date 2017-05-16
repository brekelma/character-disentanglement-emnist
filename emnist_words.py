import os
import sys
sys.path.insert(0, '../minsyn_ae')
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

path = os.getcwd()

def get_top_words(num_words = 100, folder = path, top_words_path = 'top_words.txt'):
    text_file = open(os.path.join(folder, top_words_path))
    lines = text_file.read().split('\r')
    topwords = []
    for line in lines:
        topwords.append(line)
    return np.array(topwords[:num_words])

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

def save(image, fn, save_path = 'examples'):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('right')
    try:
        os.mkdir(os.path.join(path, save_path))
    except:
        pass
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


def get_data(per_word = 500, seed = 0, data = 'train', num_words = 100, save_img = False, word_len = 3, addl_path = None, get_top_words = True, sample = 'normal'):
    if addl_path is not None:
        path = addl_path
    else:
        path = os.getcwd() 
    if type(get_top_words) is np.ndarray:
        top_words = get_top_words
    elif get_top_words:
        top_words = get_top_words(num_words = num_words, folder = path)
    else:
        print 'please enter numpy array or set get_top_words = True'
    np.random.seed(seed)
    letters_dict = read_data(dataset = data)
    original_dim = word_len*letters_dict[1][1].shape[0]*letters_dict[1][1].shape[1]
    if type(per_word) is np.ndarray:
        word_imgs = np.zeros((np.sum(per_word), original_dim))
        word_labels = np.chararray((np.sum(per_word)))
    else:
        word_imgs = np.zeros((per_word*len(top_words), original_dim))
        word_labels = np.chararray((per_word*len(top_words)))
        #word_imgs = {x:[] for x n range(len(top_words))} 
    img_count = 0
    for word_idx in range(len(top_words)):
        size = 1
        letter_idxs = []
        for lettr in top_words[word_idx]:
            letter_idx = string.lowercase.index(lettr)+1
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

        for m in range(per_word_):
        # TODO: same letter twice in word gets same sample of handwritten letter? or make both upper/lower case? 
            if m == 0 or not (sample == 'uniform'):
                i = np.random.randint(0, len(letters_dict[letter_idxs[0]])-1)
                j = np.random.randint(0, len(letters_dict[letter_idxs[1]])-1)
                k = np.random.randint(0, len(letters_dict[letter_idxs[2]])-1)
         
            temp = np.concatenate((letters_dict[letter_idxs[0]][i], letters_dict[letter_idxs[1]][j]), axis = 0)
            temp = np.concatenate((temp, letters_dict[letter_idxs[2]][k]), axis = 0)

            word_imgs[img_count] = np.reshape(temp.T, (original_dim))
            word_labels[img_count] = top_words[word_idx]

            if save_img is True:
                fn = top_words[word_idx]+'_'+ str(m)+ '.pdf'
                #show(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0]))
                save(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0])), fn)

            img_count = img_count + 1


    return word_imgs, word_labels


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

if __name__ == "__main__":
    #pw = 500
    tw = 8
    unique_seq = 8
    samples = 8000
    dataset = 'mnist'
    data_type = 'normal'

    if dataset == 'mnist':
        x_train, x_test, y_train, y_test = utilities.mnist_data()
        word_imgs = np.array(x_train[:samples, :])
        per_word = np.array([1]*10)
    else:
        sample = 'uniform'
        words = np.array(['jox', 'jix', 'box', 'bix', 'jog', 'jig', 'bog', 'big'])
        #prob = np.array([.125, .125, .125, .125, .125, .125, .125, .125])
        prob = np.array([.02, .05, .18, .01, .24, .15, .05, .3])
        if np.all(prob == prob[0]):
            dist = 'uniform'
        else:
            dist = 'nonunif'
        #prob = np.array([.02, .05, .18, .01, .24, .15, .05, .3])
        per_word = np.array(np.multiply(samples,prob)).astype(int)
        try:
            num_words = words.shape[0]
        except:
            num_words = 100

        if data_type == 'seq':
            word_imgs, word_labels = get_seq_data(total_words = tw, save_img = True)
        else:
            word_imgs, word_labels = get_data(per_word = per_word, seed = 0, num_words = 8, save_img = False, sample = sample, get_top_words = words)
            #word_imgs, word_labels = get_data(per_word = pw, seed = 0, num_words = 100, save_img = True, sample = 'uniform')
        print word_imgs.shape

    for n in [10]:
        print 'running ICA with ' + str(n) + ' components'
        ica = run_ica(word_imgs, n_components = n, max_iter = 50000)
        infomax, raw = run_infomax(word_imgs, n_components = n)
        #change to directly handle ica, infomax
        for method in ['infomax', 'ica']:
            if method == 'ica':
                A_ = ica.mixing_
                W_ = ica.components_ #unmixing matrix from x to s 
                print 'ica A: ', A_.shape
                print 'ica W: ', W_.shape
            else:
                A_ = infomax.get_components()
                W_ = infomax.unmixing_matrix_.T
                print 'infomax A: ', A_.shape
                print 'infomax W: ', W_.shape
                print 'raw size: ', np.array(raw[:,:]).shape
                print 'info: ', infomax.apply(raw)
                data, times = raw[:,:]
                reconstructions = data.T
                print 'recon ', reconstructions.shape
                #data, times = infomax.get_sources(raw)[:,:]
                #print 'get sources shape: ', data.T.shape
                #print np.array(infomax.get_sources(raw)).shape
                #raw pick channels
            #A_  = ica.mixing_matrix_
            if dataset == 'mnist':
                top_path = method + '_' + dataset
            else:
                top_path =  method + '_dist_' + dist + '_' + sample + '_letters'
            try:
                os.mkdir(os.path.join(path, top_path))
            except:
                pass
            for i in range(A_.shape[1]):
                if method == 'ica':
                    if dataset =='mnist':
                        component = np.reshape(np.abs(A_[:,i]),(28, 28))
                    else:
                        component = np.reshape(np.abs(A_[:,i]),(28,84))
                    method_path = os.path.join(top_path, 'ica components n='+str(n))
                    try:
                        os.mkdir(os.path.join(path, method_path))
                    except:
                        pass
                     
                else:
                    if dataset =='mnist':
                        component = np.reshape(np.abs(A_[:,i]),(28, 28))
                    else:
                        #what to do here?  CURRENTLY ERRORS
                        component = np.abs(A_[:,i]).reshape((28,84))

                    method_path = os.path.join(top_path, 'infomax components n='+str(n))
                    try:
                       os.mkdir(os.path.join(path, method_path))
                    except:
                       pass
                save(component, 'component_'+str(i)+'_'+str(n)+'.pdf', save_path = method_path)  
                
            if type(per_word) is np.ndarray:
                per_word_array = per_word
            else:
                per_word_array = np.array([per_word]*num_words)  

            for i in range(per_word_array.shape[0]):  
                idx = np.sum(per_word_array[:i])
                #print method, ': ', idx
                if method == 'ica':
                    #i counts in perword array (+1, cumulative sum)
                    if dataset == 'mnist':
                        recon_img = ica.inverse_transform(ica.transform(word_imgs[idx,:].reshape(1, -1))).reshape((28,28))
                    else:
                        recon_img = ica.inverse_transform(ica.transform(word_imgs[idx,:].reshape(1, -1))).reshape((28,84))
                    save(recon_img, 'recon_'+str(idx)+'_'+str(n)+'.pdf', save_path = method_path)
                else:
                    if dataset == 'mnist':
                        recon_img = reconstructions[idx,:].reshape((28,28))
                    else:
                        recon_img = reconstructions[idx,:].reshape((28,84))
                    save(recon_img, 'recon_'+str(idx)+'_'+str(n)+'.pdf', save_path = method_path)

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
        #         save(np.concatenate((recon, true), axis=1), 'recon_'+str(j)+'_'+str(n)+'.pdf', save_path = ica_path)
        # else:
        #     for j in range(100):
        #         recon = np.reshape(ica.inverse_transform(ica.transform(np.reshape(word_imgs[j*pw, :],(1,-1)))), (28,84))
        #         true = np.reshape(word_imgs[j*pw, :], (28,84))
        #         ica_path = os.path.join(top_path, 'recon n='+str(n))
        #         try:
        #             os.mkdir(os.path.join(path, top_path))
        #         except:
        #             pass
        #         save(np.concatenate((recon, true), axis=1), 'recon_'+str(j)+'_'+str(n)+'.pdf', save_path = ica_path)

    #plt.plot(S_)

    #cPickle.dump(S_, open('s_.pickle', 'wb'))
    #cPickle.dump(word_imgs, open('emnist_word_data.pickle', 'wb'))
