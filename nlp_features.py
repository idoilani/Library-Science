import collections
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import json

import gensim
from gensim import corpora

# NLP imports
import nltk
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

PROJECT_DIR = r'/Users/Gal/Documents/Repositories/Workshop-in-Data-Science'

def count_words_in_dataframe(df, column):
    '''
    counts words in the column title
    :param df: data_frame which contains a text column
    :param columns: text columns - list of columns we are going to change
    :return: dict
    '''

    rows = df[column]
    D = collections.defaultdict(lambda : 0)
    for row in rows:
        row = row.lower()
        row = row.replace(".", "")
        row = row.replace("\r\n", "")
        row = row.replace("?", "")
        row = row.replace(",", "")
        row = row.replace("=", "")
        row = row.replace("+", "")
        row = row.replace(":", " ")
        row = row.replace("-", "")
        row = row.replace("<p>", "")
        row = row.replace("{", "")
        row = row.replace("}", "")
        row = row.replace(")", "")
        row = row.replace("(", "")
        row = row.replace(";", " ")
        row = row.replace("&lt", "")
        row = row.replace("<a", "")
        row = row.replace("|", "")
        row = row.replace('"', " ")
        row = row.replace('&gt', "")
        row = row.replace('</p>', "")
        row = row.replace('%', "")
        row = row.replace('<code>', "")
        row = row.replace('</code>', "")

        words = row.split(' ')
        for word in words:
            if (word.lower() not in stopWords) and word != '':
                D[word] += 1
    return D


def remove_irrelevant_words(df, D, orig_col='qus', new_col='clean_body'):
    '''
    D is dictionary contains all word seen in the dataframe (can be generated from count_words_in_dataframe)
    :param df: dataframe to be cleaned
    :param D: dict which its keys are the words seen in text and values are the frequncies
    :return: the clean data_frame
    '''
    legal_words = set(D.keys())
    def clean_column(col):
        col = col.lower()
        col = ' '.join([x for x in col.split(" ") if x in legal_words])
        return col

    df[new_col] = df[orig_col].apply(clean_column)
    return df


def my_lda(df, column_name="clean_body", number_of_topics=10, lda_model_name="my_lda_model_qus_10_topics"):
    '''
    takes a data_frame and build an lda model for it
    :param df: ans\qus dataframe
    :return: an lda model for the relevant column of the dataframe
    '''
    docs = df[column_name]
    docs = [x.split(" ") for x in docs if type(x) != float]
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]

    # building models - might take a while
    lda = gensim.models.ldamodel.LdaModel
    lda_model = lda(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary, passes=50)

    # saving model
    lda_model.save(lda_model_name)
    return lda_model


def apply_gensim_model_on_df(gensim_model, df, column='clean_body', is_ans=False):
    '''
    :param gensim_model: gensim model generated for column
    :param df:
    :param column:
    :return: data_frame with more columns, each new column assigned to topic, the value is how much the row belongs to
    that topic
    '''

    # suffix
    suffix = "ans"
    if is_ans == False:
        suffix = "qus"

    # creates new columns, a column for each topic
    num_topics = gensim_model.num_topics
    for topic in range(num_topics):
        df["topic_"+str(topic)+"_"+suffix] = 0

    docs = df[column]
    docs = [x.split(" ") for x in docs if type(x) != float]
    dictionary = corpora.Dictionary(docs)

    # fill columns
    # bows = []
    for i, row in df.iterrows():
        if type(row[column]) != float:
            bow = dictionary.doc2bow(row[column].split(" "))
            t = gensim_model.get_document_topics(bow)
            for j in t:
                df.loc[i, "topic_"+str(j[0])+"_"+suffix] = j[1]
    return df


# uses for research - please see usage in notebook
def find_correlations_in_topics(df):
    ans_topics = ['topic_0_ans', 'topic_1_ans', 'topic_2_ans', 'topic_3_ans','topic_4_ans', 'topic_5_ans',
                  'topic_6_ans', 'topic_7_ans', 'topic_8_ans', 'topic_9_ans']
    qus_topics = ['topic_0_qus', 'topic_1_qus','topic_2_qus', 'topic_3_qus', 'topic_4_qus', 'topic_5_qus',
                  'topic_6_qus', 'topic_7_qus', 'topic_8_qus', 'topic_9_qus']

    D = collections.defaultdict(lambda: 0)
    for i, row in df.iterrows():
        print row[ans_topics]
        ans_index = np.argmax(list(row[ans_topics]))
        qus_index = np.argmax(list(row[qus_topics]))
        print ans_index , qus_index
        D[(ans_index, qus_index)] += 1

    return D


if __name__ == '__main__':

    # In this section we looked for correlations in Accepted\Rejected answers
    train = pd.read_csv("C:\\Users\\Gal\\Documents\\Library-Science\\combined_train.csv")
    test = pd.read_csv("C:\\Users\\Gal\\Documents\\Library-Science\\combined_test.csv")

    print "train and test are loaded"

    df = pd.concat([train, test])
    df_accept = df[df['IsAcceptedAnswer'] == 1]
    df_reject = df[df['IsAcceptedAnswer'] == 0]

    D_accept = find_correlations_in_topics(df_accept)
    with open('D_accept.json', 'w') as f:
        json.dump(D_accept, f)

    D_reject = find_correlations_in_topics(df_reject)
    with open('D_reject.json', 'w') as f:
        json.dump(D_reject, f)
