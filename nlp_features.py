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


# TODO - maybe we can do the cleaning better - for example all numbers become to NUM
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

# TODO - change function name to remove_irrelevant_words
def remove_irrelevant_words(df, D, orig_col='qus', new_col='clean_body'):
    '''
    D is dictionary contains all word seen in the dataframe (can be generated from count_words_in_dataframe)
    :param df: dataframe to be cleaned
    :param D: dict which its keys are the words seen in text and values are the frequncies (NOT RELEVANT NOW)
    :return: the clean data_frame
    '''
    legal_words = set(D.keys())
    def clean_question(qus):
        qus = qus.lower()
        qus = ' '.join([x for x in qus.split(" ") if x in legal_words])
        return qus

    df[new_col] = df[orig_col].apply(clean_question)
    return df


# TODO - add more documentation on this function
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



# TODO - for notebook
def find_correlations_in_topics(df):
    ans_topics = ['topic_0_qus_Ans', 'topic_1_qus_Ans', 'topic_2_qus_Ans', 'topic_3_qus_Ans',
                'topic_4_qus_Ans', 'topic_5_qus_Ans', 'topic_6_qus_Ans', 'topic_7_qus_Ans', 'topic_8_qus_Ans',
                'topic_9_qus_Ans']
    qus_topics = ['topic_0_qus_Qus', 'topic_1_qus_Qus',
                'topic_2_qus_Qus', 'topic_3_qus_Qus', 'topic_4_qus_Qus', 'topic_5_qus_Qus', 'topic_6_qus_Qus',
                'topic_7_qus_Qus', 'topic_8_qus_Qus', 'topic_9_qus_Qus']

    D = collections.defaultdict(lambda:0)
    for i, row in df.iterrows():
        ans_index = np.argmax(row[ans_topics])
        qus_index = np.argmax(row[qus_topics])
        D[(ans_index, qus_index)] += 1

    return D

if __name__ == '__main__':
    project_dir = '/Users/Gal/Downloads/rquestions'
    #ans = pd.read_csv(project_dir + '/Clean_Answers.csv')

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

    print D_accept.values()
    print D_reject.values()

    #qus = pd.read_csv(project_dir + '/Clean_Questions.csv')
    #qus = qus
    #tags = pd.read_csv(project_dir + '/Clean_Tags.csv')

    # D_title = count_words_in_dataframe(qus, 'Title')
    #D_body = count_words_in_dataframe(ans, 'Body')

    #clean_ans = clean_data(ans, D_body, 'Body', 'clean_body')
    #clean_ans.to_csv('C:\\Users\\Gal\\Documents\\Library-Science\\clean_answers_df.csv')

    '''
    Documentation for later:
    cleaning questions body:

    qus = pd.read_csv(project_dir + '/Clean_Questions.csv')
    D_body = count_words_in_dataframe(qus, 'Body')
    clean_ans = clean_data(qus, D_body, 'Body', 'clean_body')

    cleaning answers body:

    ans = pd.read_csv(project_dir + '/Clean_Answers.csv')
    D_body = count_words_in_dataframe(ans, 'Body')
    clean_ans = clean_data(ans, D_body, 'Body', 'clean_body')
    '''


    #print sorted(D_body.iteritems(), key=lambda (k, v): v, reverse=True)[:200]
    # print sorted(D_title.iteritems(), key=lambda (k, v): v, reverse=True)[:200]