# import librarys
import numpy as np # for data preprocessing
import pandas as pd # for load excel data
import bitermplus as btm # for topic modeling
import matplotlib.pyplot as plt # for visualization result
import seaborn as sns # for visualization result
from wordcloud import WordCloud # for visualization result as a wordcloud
import platform # for check the OS

# for ignore the warning message
import warnings
warnings.filterwarnings("ignore")

if platform.system() in ["Windows", "Linux"]:
    plt.rcParams["font.family"] = "Malgun Gothic"
    font_path = "malgun"

elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
    font_path = "AppleGothic"

else:
    print("Unsupported OS.")

plt.rcParams["axes.unicode_minus"] = False

def find_proper_t(X:np.ndarray, vocabulary:np.ndarray, start:int, end:int):
    """find proper k value for hyperparameter tunning"""
    for i in range(start, end+1):
        model = btm.BTM(X, vocabulary, seed=42, T=i, M=20, alpha=50/i, beta=0.01)
        coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
        
        # initial value setup
        if i == start:
            proper_k = start
            tmp = np.mean(coherence)
        
        # print it out
        print('==== Coherence : k = {} ===='.format(i))
        print("\n")
        print('Average: {}'.format(np.mean(coherence)))
        print("\n")
        print('Per Topic:{}'.format(coherence))
        print("\n")
        print("\n")
        
        # update k
        if tmp < np.mean(coherence):
            proper_k = i
            tmp = np.mean(coherence)
    return proper_k

# data load
# The data is related to Handong University, 
# which was reported in major Korean daily newspapers from January 1995 to September 2024.
df = pd.read_excel("data/NewsResult_19950101-20240930.xlsx", engine="openpyxl")

# Preprocessing text data
# In this case, I use headline data for BTM
# Becauese BTM is specialized in short texts.
headline = df['제목'].str.strip().tolist()

# Obtaining terms frequency in a sparse matrix and corpus vocabulary
X, vocabulary, vocab_dict = btm.get_words_freqs(headline)
tf = np.array(X.sum(axis=0)).ravel()

# Vectorizing documents
docs_vec = btm.get_vectorized_docs(headline, vocabulary)
docs_lens = list(map(len, docs_vec))

# Generating biterms
biterms = btm.get_biterms(docs_vec)

# find the proper T
topic_num = find_proper_t(X, vocabulary, 3, 10)

# train the model
model = btm.BTM(X, vocabulary, seed=42, T=topic_num, M=20, alpha=50/7, beta=0.01)
iterations = 2000
model.fit(biterms, iterations)

# extract the topics-words distribution
topic_word_dist = model.matrix_topics_words_

# print the topic's top word
def print_topics(topic_word_dist, vocab, n_top_words=10):
    for i, topic_dist in enumerate(topic_word_dist):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words-1:-1]
        print(f"Topic {i}: {' '.join(topic_words)}")

print_topics(topic_word_dist, vocabulary, n_top_words=10)

# make the wordcloud for show top word's frequency.
def draw_word_clouds(topic_word_dist, vocab):
    for i, topic_dist in enumerate(topic_word_dist):
        plt.figure(figsize=(8, 6))
        freq = {vocab[j]: topic_dist[j] for j in range(len(vocab))}
        wordcloud = WordCloud(background_color='white', font_path = "AppleGothic").generate_from_frequencies(freq)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {i}')
        plt.show()

draw_word_clouds(topic_word_dist, vocabulary)

# make the histogram for show topic distribution
doc_topics = model.transform(docs_vec)
topic_proportions = doc_topics.mean(axis=0)

plt.figure(figsize=(8, 6))
sns.barplot(x=[f'Topic {i}' for i in range(len(topic_proportions))], y=topic_proportions)
plt.title('topic distribution')
plt.ylabel('ratio')
plt.show()