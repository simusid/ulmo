from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from joblib import Parallel, delayed
import random
from tqdm import tqdm
import os
from scipy.io import wavfile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pickle as pkl
 
from sklearn.decomposition import PCA

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim
        )
        self.pos_emb = layers.Embedding(
            input_dim=sequence_length, output_dim=embedding_dim
        )

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        
class LAM():
    def __init__(self, x,y, epochs = 12, num_heads=4, num_transformer_blocks=2, 
                ff_dim=512, sequence_length=50, embedding_dim=128, tokenizer_vocab_size=200):
        # Parameters
        # TODO - don't hard code sequence length
        self.vocab_size = tokenizer_vocab_size         # Size of the token vocabulary
        self.sequence_length = sequence_length         # Length of input sequences  originally 20
        self.embedding_dim = embedding_dim             # Dimension of token embeddings originally 128
        self.num_heads = num_heads                     # Number of attention heads   (originally 4)
        self.ff_dim = ff_dim                           # Dimension of feedforward network  (originally 512)
        self.dropout_rate = 0.2                        # Dropout rate (orieginally .1)
        self.num_transformer_blocks = num_transformer_blocks  # Number of transformer blocks originally 2
        self.batch_size = 256                          # Batch size for training
        self.epochs =  epochs                          # Number of training epochs
        self.x_train = x
        self.y_train = y

    def transformer_block(self, inputs, embedding_dim, num_heads, ff_dim, dropout_rate):
        # Multi-head Self-Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(inputs, inputs)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
        # Feedforward Network
        ffn = layers.Dense(ff_dim, activation="gelu")(attention_output)
        ffn = layers.Dense(embedding_dim)(ffn)
        ffn_output = layers.Dropout(dropout_rate)(ffn)
        sequence_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        return sequence_output

    def build(self):
        inputs = keras.Input(shape=(self.sequence_length,))
        # Embedding layer with positional encoding
        embedding_layer = TokenAndPositionEmbedding(self.sequence_length, self.vocab_size, self.embedding_dim)
        x = embedding_layer(inputs)
        
        # Stack of transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(x, self.embedding_dim, self.num_heads, self.ff_dim, self.dropout_rate)
        
        # Use the representation of the last token for prediction
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Lambda(lambda x: x[:, -1, :])(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.ff_dim, activation="gelu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        x = layers.Dense(self.vocab_size)(x)
        outputs = layers.Softmax()(x)
        
        # Create the model
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def train(self, learning_rate=0.0003):
        adam = Adam(learning_rate)
        es = EarlyStopping(patience=4)
        
        self.model.compile( optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # TODO - Fix data source, and Train/test split
        
        x_all= self.x_train
        y_all =self.y_train

        idx = int(x_all.shape[0]*.9)
        x_train = x_all[:idx,:]
        y_train = y_all[:idx]
        x_test  = x_all[idx:,:]
        y_test  = y_all[idx:]
        self.history = self.model.fit(x_train, y_train, epochs = self.epochs, callbacks=[es],
                                    batch_size = self.batch_size, validation_split=.15, )
        self.model.evaluate(x_test, y_test)

    def evaluate(self):
        pass

# takes a time series wav file and produces a set of mel spectrograms
class MultiChannelGrams():
    def __init__(self, fname, channel = 0, target_sr= 8000, n_mels=256, gram_width=32):
        self.fname=fname
        self.gram_width=gram_width
        self.n_mels = n_mels
        self.target_sr = target_sr
        self.channel = channel
        self.makeGrams()
        
    def xToMelSpectrograms(self, x):
        epsilon = np.finfo(np.float64).eps
        grams = []
        y     = librosa.feature.melspectrogram(y=x, n_mels=self.n_mels, hop_length=16)
        y     = y + epsilon
        Sxx = np.log(np.abs(y))
        Sxx = Sxx - Sxx.min()
        Sxx = Sxx/Sxx.max()
        for i in range(0, Sxx.shape[1] - self.gram_width, self.gram_width//2):
            grams.append(Sxx[:, i:i+self.gram_width])
        return grams, Sxx
    
    # passed a multichannel time series file
    # populates member vars with a list of spectrgrams
    def makeGrams(self):
        if(self.fname.endswith(".mp3")):
            x, sr = librosa.load(self.fname, sr=self.target_sr)
            if(len(x.shape)==1):
                x = np.expand_dims(x, axis=1)
        else:
            sr, x = wavfile.read(self.fname)    # default padres samping rate is 80,000
        # process only one channel
        x = x[:, self.channel]
        x= x-x.min()
        x= x/x.max()
        y = librosa.resample(y=x, orig_sr=sr, target_sr=self.target_sr)
        self.grams, self.Sxx = self.xToMelSpectrograms(y)


class LAM_KMeans():
    def __init__(self, grams=[], vocabulary_size=100, reduce_dims=False, umap_components=256):
        self.vocabulary_size= vocabulary_size
        self.grams = grams
        self.batch_size= 1000
        self.UMAP_COMPONENTS=umap_components
        self.REDUCE_DIMS = reduce_dims 
        
        self.model = MiniBatchKMeans(n_clusters=self.vocabulary_size, batch_size=self.batch_size, verbose=False)
        if(grams):
            features = np.vstack([g.ravel() for g in grams])
            if(self.REDUCE_DIMS==True):
                features= self.doReduceDim(features)
            
            self.model.fit(features)
        # self.kmeans.partial_fit(features[1000:23000])
        
    def doReduceDim(self, x):
        # had to swap in PCA for UMAP because of broken numpy/numba dependencies on platform
        #mapper = UMAP(n_components=self.UMAP_COMPONENTS).fit_transform(x)
        pca = PCA(n_components=self.UMAP_COMPONENTS)
        return pca.fit_transform(x)
        
    def partial_fit(self, grams):

        features = np.vstack([g.ravel() for g in grams])
        if(self.REDUCE_DIMS==True):
            features= self.doReduceDim(features)
        self.model.partial_fit(features)

    def predict(self, grams):
        features = np.vstack([g.ravel() for g in grams])
        if(self.REDUCE_DIMS==True):
            features= self.doReduceDim(features)
        return self.model.predict(features)


class LAM_Tokenizer():
    def __init__(self,kmeans, labels, vocab_size=100):
        self.kmeans = kmeans
        self.labels = labels
        self.vocab_size= vocab_size
        self.tokenizer = Tokenizer(BPE())
        self.trained = False    
        
    def train(self):
        # IMPORTANT - Note we are training on strings of integers separated by a space
        labels_as_strings = [" ".join(map(str, l)) for l in self.labels]
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=3, show_progress=True)
        self.tokenizer.train_from_iterator(labels_as_strings, trainer=trainer)
        self.trained = True
        
    def tokenizeFile(self, f):
        if(self.trained ==False):
            print("Cannot tokenize file.  Tokenizer has not been trained")
            return 
        try:
            grams = MultiChannelGrams(f).grams    # step 1 load the file and convert to grams
            labels = self.kmeans.predict(grams)   # step 2 use trained kmeans model to map gram to label
            labels_as_string = " ".join(map(str, labels))  
            return self.tokenizer.encode(" ".join(map(str, labels_as_string))).ids  # step 3 convert labels to tokens
        except Exception as e:
            print("cannot tokenize file", f, e)
            return []