# import librarys
import numpy as np # for data preprocessing
import pandas as pd # for load excel data
import pyBigKinds as pbk # for preprocessing bigkinds text data
import tomotopy as tp # for topic modeling
from pyvis.network import Network # for visualize the CTM result

# for ignore the warning message
import warnings
warnings.filterwarnings("ignore")

def ctmmodel(df:pd.DataFrame, k:int):
    """Define CTM model """
    
    words = pbk.keyword_parser(pbk.keyword_list(df))
    model = tp.CTModel(tw=tp.TermWeight.IDF, min_cf=5, k=k)

    for k in range(len(words)):
        model.add_doc(words=words[k])

    model.train(0)

    # Since we have more than ten thousand of documents, 
    # setting the `num_beta_sample` smaller value will not cause an inaccurate result.
    model.num_beta_sample = 5
    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(model.docs), len(model.used_vocabs), model.num_words
    ))
    print('Removed Top words: ', *model.removed_top_words)
    
    # train the model
    model.train(1000, show_progress=True)
    model.summary()

    return model


def find_proper_k(df:pd.DataFrame, start:int, end:int):
    """find proper k value for hyperparameter tunning"""

    words = pbk.keyword_parser(pbk.keyword_list(df))

    for i in range(start,end+1):        
        # model setting
        mdl=tp.CTModel(tw=tp.TermWeight.IDF, min_cf=5, k=i)
        
        for k in range(len(words)):
            mdl.add_doc(words=words[k])
            
        # pre-train the model for check the coherence score
        mdl.train(50)
        
        # get the coherence score
        coh = tp.coherence.Coherence(mdl, coherence='c_v')
        
        # coherence average
        average_coherence = coh.get_score()
        # initial value setup
        if i == start:
            proper_k = start
            tmp = average_coherence
        
        # get coherence per topic
        coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
        
        # print it out
        print('==== Coherence : k = {} ===='.format(i))
        print("\n")
        print('Average: {}'.format(average_coherence))
        print("\n")
        print('Per Topic:{}'.format(coherence_per_topic))
        print("\n")
        print("\n")
        
        # update k
        if tmp < average_coherence:
            proper_k = i
            tmp = average_coherence
    return proper_k


def get_ctm_network(mdl):
    """ctm result visualization through Network"""

    g = Network(width=800, height=800, font_color="#333")
    correl = mdl.get_correlations().reshape([-1])
    correl.sort()
    top_tenth = mdl.k * (mdl.k - 1) // 10
    top_tenth = correl[-mdl.k - top_tenth]
    
    for k in range(mdl.k):
        label = "#{}".format(k)
        title= ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=6))
        print('Topic', label, title)
        g.add_node(k, label=label, title=title, shape='ellipse')
        for l, correlation in zip(range(k - 1), mdl.get_correlations(k)):
            if correlation < top_tenth: continue
            g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))
    
    g.barnes_hut(gravity=-1000, spring_length=20)
    g.show_buttons()
    g.show("topic_network.html", notebook=False)
    
    
# data load
# The data is related to Handong University, 
# which was reported in major Korean daily newspapers from January 1995 to September 2024.
df = pd.read_excel("data/NewsResult_19950101-20240930.xlsx", engine="openpyxl")

#find proper k value
proper_k = find_proper_k(df, 2,10)

# Model setting with K
mdl = ctmmodel(df, proper_k)

# get summary
mdl.summary()

# save model
get_ctm_network(mdl)