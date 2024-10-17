# import librarys
import numpy as np # for data preprocessing
import pandas as pd # for load excel data
import pyBigKinds as pbk # for preprocessing news text data
import tomotopy as tp # for topic modeling
import pyLDAvis # for visualize the LDA result

# for ignore the warning message
import warnings
warnings.filterwarnings("ignore")

def ldamodel(df:pd.DataFrame, k:int):
    """Define the LDA model """
    
    words = pbk.keyword_parser(pbk.keyword_list(df))
    model=tp.LDAModel(min_cf=5, k=k)

    for k in range(len(words)):
        model.add_doc(words=words[k])
    
    mdl.train(0)

    # print docs, vocabs and words
    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
    ))

    # train the model
    mdl.train(1000, show_progress=True)
    return model

def find_proper_k(df:pd.DataFrame, start:int, end:int):
    """find proper k value for hyperparameter tunning"""

    words = pbk.keyword_parser(pbk.keyword_list(df))

    for i in range(start,end+1):        
        # model setting
        mdl=tp.LDAModel(min_cf=5, k=i)
        
        for k in range(len(words)):
            mdl.add_doc(words=words[k])
            
        # pre-train the model for check the coherence score
        mdl.train(100)
        
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

def get_ldavis(mdl):
    """preprocessing for LDA Topic data visualization"""
    
    # get the values
    topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq

    # prepara dataset
    prepared_data = pyLDAvis.prepare(
        topic_term_dists, 
        doc_topic_dists, 
        doc_lengths, 
        vocab, 
        term_frequency,
        start_index=0, 
        sort_topics=False 
    )

    # save result
    pyLDAvis.save_html(prepared_data, 'ldavis.html')

# data load
# The data is related to Handong University, 
# which was reported in major Korean daily newspapers from January 1995 to September 2024.
df = pd.read_excel("data/NewsResult_19950101-20240930.xlsx", engine="openpyxl")

#find proper k value
proper_k = find_proper_k(df, 2,10)

# Model setting with K
mdl = ldamodel(df, proper_k)

# get summary
mdl.summary()

# save model
get_ldavis(mdl)