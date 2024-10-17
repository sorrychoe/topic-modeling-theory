# import librarys
import numpy as np # for data preprocessing
import pandas as pd # for load excel data
import pyBigKinds as pbk # for preprocessing news text data
import tomotopy as tp # for topic modeling
import matplotlib.pyplot as plt # for visualize the plot

# for ignore the warning message
import warnings
warnings.filterwarnings("ignore")

def dtmmodel(df:pd.DataFrame, k:int):
    """Define the DTM model """
    words = pbk.keyword_parser(pbk.keyword_list(df))
    t = np.max(df["시점"])+1
    model = tp.DTModel(min_cf=5, rm_top=10, t=t, k=k, seed=42)

    for i in range(len(words)):
        model.add_doc(words=words[i], timepoint=df["시점"][i])

    model.train(0)

    # print docs, vocabs and words
    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(model.docs), len(model.used_vocabs), model.num_words
    ))

    model.train(2000, show_progress=True)
    
    return model

def find_proper_k(df, start, end):
    """find proper k value for hyperparameter tunning""" 
    words = pbk.keyword_parser(pbk.keyword_list(df))
    for i in range(start,end+1):        
        # model setting
        mdl=tp.DTModel(min_cf=5, rm_top=10, k=i, t=(np.max(df["시점"])+1), seed=42)
        
        for k in range(len(words)):
            mdl.add_doc(words=words[k], timepoint=df["시점"][k])
        
        mdl.train(100)
        
        # coherence score
        print('==== Coherence : k = {} ===='.format(i))
        print("\n")
        coh = tp.coherence.Coherence(mdl, coherence="c_v")
        
        average_coh = []
        
        for j in df["시점"].unique():
            t_coh = [coh.get_score(topic_id=c, timepoint=j) for c in range(mdl.k)]
            
            print('timepoint: {}'.format(j))
            print("\n")
            print('average:{}'.format(np.mean(t_coh)))
            print("\n")
            print('Per Topic:{}'.format(t_coh))
            print("\n")
            
            average_coh.append(t_coh)
        if i == 3:
            proper_k = 3
            tmp = np.mean(average_coh)
        elif tmp < np.mean(average_coh):
            proper_k = i
            tmp = np.mean(average_coh)     

    return proper_k    

def dtm_plot(df):
    """preprocessing for DTM Topic data visualization"""
    f = plt.figure()
    plt.title('Changes in topic percentage by year', color='black')
    df.plot(ax=f.gca(),colormap='gnuplot')
    plt.xticks(np.array(range(1995,2024)),rotation=45)

    ax=f.gca()
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_ylabel('percent(%)')
    plt.show()
        
# data load
# The data is related to Handong University, 
# which was reported in major Korean daily newspapers from January 1995 to September 2024.
df = pd.read_excel("data/NewsResult_19950101-20240930.xlsx", engine="openpyxl")

# add the time stamp for DTM
df["시점"] = (round(df["일자"]/10000,0)%1995).astype(int)
df = df.sort_values("시점")
df.reset_index(drop=True, inplace=True)

#find proper k value
proper_k = find_proper_k(df, 3 ,10)

# Model setting with K
mdl = dtmmodel(df, proper_k)

# get summary
mdl.summary()

# get dtm dataframe
dtm_df = pd.DataFrame(mdl.get_count_by_topics())
dtm_df.index = df["시점"].unique() + 1995

# show result as a plot
dtm_plot(dtm_df)