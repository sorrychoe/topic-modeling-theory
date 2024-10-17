# import librarys
import pandas as pd # for load excel data
import pyBigKinds as pbk # for preprocessing news data
from sklearn.feature_extraction.text import CountVectorizer # for vectorize text data 
from konlpy.tag import Mecab # for tokenize the Korean Words
from bertopic import BERTopic # for load the BERTopic model
from hdbscan import HDBSCAN # for tunning the BERTopic model

# for ignore the warning message
import warnings
warnings.filterwarnings("ignore")

def list_to_str(words: list):
    """Function that list data change to string for text preprocessing"""
    for i in range(len(words)):
        text = ""
        for word in words[i]:
            if text == "":
                text = word
            else: 
                text = text + " " + word
        words[i] = text
    return words

class CustomTokenizer:
    """ Define the Korean Tokenizer"""
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result
    
# data load
# The data is related to Handong University, 
# which was reported in major Korean daily newspapers from January 1995 to September 2024.
df = pd.read_excel("data/NewsResult_19950101-20240930.xlsx", engine="openpyxl")

# add the time stamp for ToT
df["시점"] = (round(df["일자"]/10000,0)%1995).astype(int)
df = df.sort_values("시점")
df.reset_index(drop=True, inplace=True)

# text Preprocessing 
words = pbk.keyword_parser(pbk.keyword_list(df))
words = list_to_str(words)
timestamp = df["시점"].tolist()

# Define tokenizer & Vectorizer
custom_tokenizer = CustomTokenizer(Mecab())
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)


hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', prediction_data=True) 

# Define Topic Model for ToT
topic_model = BERTopic(
    embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", 
    vectorizer_model=vectorizer,
    hdbscan_model=hdbscan_model,
    top_n_words=20,
    min_topic_size=3,
    calculate_probabilities=True
)
# fit the data
topics, probs = topic_model.fit_transform(words)

# get the ToT result
topics_over_time = topic_model.topics_over_time(words, timestamp)

# show dataframe of ToT Result
topics_over_time.head(10)

#show line plot of ToT result
visual = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=30)

# save the result
visual.write_html("view/tot.html")