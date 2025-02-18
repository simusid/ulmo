
from lam import *
from glob import glob
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm 
from argparse import ArgumentParser
from pathlib import Path 
from experiments import ExperimentManager

parser = ArgumentParser()
parser.add_argument("--source", type=str, help="Path to the audio files")   
parser.add_argument("--channel", type=int, default=0, help="Channel to use for audio")
parser.add_argument("--target_sr", type=int, default=3000, help="Target sample rate")
parser.add_argument("--n_mels", type=int, default=256, help="Number of mel bins")    
parser.add_argument("--gram_width", type=int, default=32, help="Width of the spectrogram")  
parser.add_argument("--kmeans_vocabulary_size", default=200, type=int, help="Size of the kmeans vocabulary")
parser.add_argument("--reduce_dims", type=bool, default=0, help="Reduce dimensions")
parser.add_argument("--umap_components", type=int,default=512, help="UMAP components")
parser.add_argument("--tokenizer_vocabulary_size", default=150, type=int, help="Tokenizer vocabulary size")
parser.add_argument("--embedding_dim", type=int,default=128, help="Embedding dimension")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
parser.add_argument("--num_transformer_blocks", default=2, type=int, help="Number of transformer blocks")
parser.add_argument("--ff_dim", default=512, type=int, help="Feed forward dimension")
parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length")
parser.add_argument("--max_files", default=1000, type=int, help="Maximum number of files to process")
args = parser.parse_args()

manager = ExperimentManager()  # create new directory for experiment 
exp_path = manager.new()
note = input("Enter a note for this experiment: ")
logging.basicConfig(level=logging.INFO, filename=f'{exp_path}/lam.log')
logger = logging.getLogger()
logger.info("Starting LAM")
logger.info(note)
logger.info(f"KMeans vocabulary size: {args.kmeans_vocabulary_size}")
logger.info(f"Tokenizer vocabulary size: {args.tokenizer_vocabulary_size}")
logger.info(f"sequence length: {args.sequence_length}")


path = Path(args.source)   # 
fnames = list(path.rglob("*.mp3") )+ list(path.rglob("*.wav"))
fnames = [str(f) for f in fnames]
random.shuffle(fnames)

logger.info(f"Found {len(fnames)} audio files")
fnames = random.sample(fnames, args.max_files)
logger.info(f"limited to: {len(fnames)} audio files")

# create grams from the first file and fit a kmeans model 
grams = MultiChannelGrams(fnames[0], args.channel, args.target_sr, args.n_mels, args.gram_width)   
assert len(grams.grams)>0
kmeans = LAM_KMeans(grams.grams, vocabulary_size=args.kmeans_vocabulary_size, 
    reduce_dims=args.reduce_dims, 
    umap_components=args.umap_components) 

# now load and train the rest of the files with partial_fit
for f in tqdm(fnames[1:]):
    try:
        grams=MultiChannelGrams(f, args.channel, args.target_sr, args.n_mels, args.gram_width).grams
        kmeans.partial_fit(grams)
        #augment 10% higher
        aug_sr = args.target_sr*1.1
        grams=MultiChannelGrams(f, args.channel, args.target_sr, args.n_mels, args.gram_width).grams
        kmeans.partial_fit(grams)
        #augment 10% lower
        aug_sr = args.target_sr*.9
        grams=MultiChannelGrams(f, args.channel, args.target_sr, args.n_mels, args.gram_width).grams
        kmeans.partial_fit(grams)
    except Exception as e:
        logger.error(f"Error processing {f}: {e}")
        continue
 
labels = []
for f in tqdm(fnames):
    grams = MultiChannelGrams(f, args.channel, args.target_sr, args.n_mels, args.gram_width).grams
    labels.append(kmeans.predict(grams))

logger.info("KMeans clustering complete")
tokenizer = LAM_Tokenizer(kmeans, labels, args.tokenizer_vocabulary_size)
logger.info("Tokenizer training complete")
tokenizer.train()
 
tokens = [tokenizer.tokenizeFile(f) for f in tqdm(fnames)]
logger.info("Tokenization complete")

x = []
y = []
for t in tokens:
    N = args.sequence_length
    for i in range(0, len(t)-N-1, N):
        x.append(t[i:i+N])
        y.append( t[i+N])
assert len(x)==len(y)
assert len(x)>0

x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)
lam = LAM(x, y, sequence_length=args.sequence_length)
lam.build()
lam.train()

# this belongs in an Assessments class
def histogram_metric(data, bins=10):
    counts, bin_edges = np.histogram(data, bins='auto')
    most_values = np.max(counts)
    total_counts = np.sum(counts)
    metric = most_values / total_counts if total_counts > 0 else 0
    return metric

plt.plot(lam.history.history['loss'])
plt.title("Loss")
plt.grid()
plt.savefig(exp_path + "/loss.png")

plt.hist(y, bins=tokenizer.vocab_size)
plt.title(f"Next Token Distribution - {histogram_metric(y)}")
plt.grid()
plt.savefig(exp_path + "/next_token.png")