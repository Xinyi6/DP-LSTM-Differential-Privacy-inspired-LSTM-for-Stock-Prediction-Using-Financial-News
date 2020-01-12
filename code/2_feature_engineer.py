#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
df = pd.read_csv('us_financial_news_articles_2018.csv',index_col=0)


# In[11]:


df.head()


# ## only 2 rows of missing data, we can fill it with anything

# In[12]:


df[df.isnull().any(axis=1)]


# In[13]:


df_missing_percentage=df.isnull().sum()/df.shape[0] *100


# In[14]:


df_missing_percentage


# In[15]:


df=df.fillna('missing')


# In[16]:


df.shape


# # Feature Engineer

# ## TF-IDF: term frequency–inverse document frequency
# ## Bag of Words

# In[17]:


def create_tfidf(df, feature_column, max_feature_size):
    from sklearn.feature_extraction import text
    from sklearn.feature_extraction.text import TfidfVectorizer

    my_stop_words = text.ENGLISH_STOP_WORDS.union(["ap1", "00", "000", "0", "561", "190", "09", "24", "2017","2018", "000 00", "2018",
                               "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
                               "ag", "ap3", "000 00 ap3", "ap2", "00 ap2", "000 00 ap2", "10", "00 ap1",
                               "000 00 ap1", "oct 2018", "000 000", "000 000 00", "october 2018", "10 2018",
                               "11 2018", "november 2018", "12 2018", "december 2018"


                               ])
    #my_stop_words=my_stop_words.difference(["call"])

    # initilize 
    tfidf_vec = TfidfVectorizer(sublinear_tf=True,min_df=2,norm='l2',encoding="latin-1",ngram_range=(1,3),stop_words=my_stop_words,max_features=max_feature_size)
    features = tfidf_vec.fit_transform(df[feature_column]).toarray()
    return features, tfidf_vec


# In[23]:


def create_bow(df, feature_column, max_feature_size):
    from sklearn.feature_extraction import text
    from sklearn.feature_extraction.text import CountVectorizer

    my_stop_words = text.ENGLISH_STOP_WORDS.union(["ap1", "00", "000", "0", "561", "190", "09", "24", "2017","2018", "000 00", "2018",
                               "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
                               "ag", "ap3", "000 00 ap3", "ap2", "00 ap2", "000 00 ap2", "10", "00 ap1",
                               "000 00 ap1", "oct 2018", "000 000", "000 000 00", "october 2018", "10 2018",
                               "11 2018", "november 2018", "12 2018", "december 2018"


                               ])
    #my_stop_words=my_stop_words.difference(["call"])

    # initilize 
    Counter_vec = CountVectorizer(encoding="latin-1",ngram_range=(1,3),stop_words=my_stop_words,max_features = max_feature_size)
    features = Counter_vec.fit_transform(df[feature_column]).toarray()
    return features, Counter_vec


# In[18]:


def create_words_frequency(features, features_name):
    features_df=pd.DataFrame(features)
    features_df.columns=features_name
    sorted_features = features_df.sum(axis=0).sort_values(ascending=False)
    sorted_features=pd.DataFrame(sorted_features)
    sorted_features=sorted_features.reset_index()
    sorted_features.columns=['Top Words','Counts']
    return sorted_features


# In[ ]:





# In[19]:


features_tfidf, tfidf_vec = create_tfidf(df, feature_column = 'title',max_feature_size=5000)


# In[20]:


features_tfidf_names=tfidf_vec.get_feature_names()


# In[21]:


tfidf_sorted_table= create_words_frequency(features_tfidf, features_tfidf_names)


# In[22]:


tfidf_sorted_table.head(20)


# In[24]:


features_bow, bow_vec = create_bow(df, feature_column = 'title',max_feature_size=5000)


# In[25]:


features_bow_names=bow_vec.get_feature_names()


# In[27]:


bow_sorted_table= create_words_frequency(features_bow, features_bow_names)


# In[28]:


bow_sorted_table.head(20)


# ## Sentiment Score
# ### nltk.sentiment.vader package [medium_blog](https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f).
# ### nltk documentation[nltk_blog](https://www.nltk.org/api/nltk.sentiment.html)

# In[29]:


#SentimentIntensityAnalyzer package
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[30]:


get_ipython().run_cell_magic('time', '', 'title_score = [sid.polarity_scores(sent) for sent in df.title]')


# In[31]:


len(title_score)


# In[32]:


df.title[0:10].values


# In[33]:


title_score[0:10]


# In[37]:


compound=[]
neg=[]
neu=[]
pos=[]

for i in range(len(title_score)):
    compound.append(title_score[i]['compound'])
    neg.append(title_score[i]['neg'])
    neu.append(title_score[i]['neu'])
    pos.append(title_score[i]['pos'])


# In[39]:


len(compound)


# In[40]:


len(neg)


# In[42]:


len(neu)


# In[38]:


len(pos)


# In[43]:


df['compound'] = compound
df['neg'] = neg
df['neu'] = neu
df['pos'] = pos


# In[44]:


df.head()


# In[46]:


#df.to_csv('us_financial_news_articles_2018_with_sentiment.csv')

