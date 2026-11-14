# import librarys
import pandas as pd # for load excel data
import pyBigKinds as pbk # for preprocessing news data
from sklearn.feature_extraction.text import CountVectorizer # for vectorize text data
from konlpy.tag import Mecab # for tokenize the Korean Words
from bertopic.vectorizers import ClassTfidfTransformer # for get c-TF-IDF value
from bertopic import BERTopic # for load the BERTopic model
from hdbscan import HDBSCAN # for tunning the BERTopic model
from umap import UMAP # for tunning the BERTopic model

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
df = pd.read_excel("../data/NewsResult_19950101-20240930.xlsx", engine="openpyxl")

# add the time stamp for BERTopic's topics_over_time
df["시점"] = (round(df["일자"]/10000,0)).astype(int)
df = df.sort_values("시점")
df.reset_index(drop=True, inplace=True)

# text Preprocessing
words = pbk.keyword_parser(pbk.keyword_list(df))
words = list_to_str(words)
timestamp = df["시점"].tolist()

# Define tokenizer & Vectorizer
custom_tokenizer = CustomTokenizer(Mecab())
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

# set the HDBSCAN Model for tunning BERTopic model
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=False, min_cluster_size=20)

# Define Topic Model for BERTopic
topic_model = BERTopic(
    embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
    vectorizer_model=vectorizer,
    hdbscan_model=hdbscan_model,
    ctfidf_model=ctfidf_model,
    top_n_words=20,
    min_topic_size=20,
    verbose=True
)

# fit the data
topics, probs = topic_model.fit_transform(words)

# show topic dataframe
print(topic_model.get_topic_info())

# show 2d visualized topic like LDAvis
BERTopic_vis = topic_model.visualize_topics()

# To visualize the hierarchy of topics
hierachy_vis = topic_model.visualize_hierarchy()

# To visualize the correlation of topics
heatmap_vis = topic_model.visualize_heatmap()

# To visualize the ranking of terms per topic
barchart_vis = topic_model.visualize_barchart(top_n_topics=20)

# To visualize the Term score per topic
rank_vis = topic_model.visualize_term_rank()

# get the topics_over_time result
topics_over_time = topic_model.topics_over_time(words, timestamp)

# show dataframe of topics_over_time result
print(topics_over_time.head(20))

#show line plot of topics_over_time result
tot_vis = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=30)

# save the result
BERTopic_vis.write_html("../view/bertopic_vis.html")
hierachy_vis.write_html("../view/hierachy_vis.html")
heatmap_vis.write_html("../view/heatmap_vis.html")
barchart_vis.write_html("../view/barchart_vis.html")
rank_vis.write_html("../view/rank_vis.html")
tot_vis.write_html("../view/tot_vis.html")
