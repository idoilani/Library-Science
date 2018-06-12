import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


import gensim
Lda = gensim.models.ldamodel.LdaModel
from gensim import corpora

def combine_qus_and_ans(qus_df, ans_df, user_df):
    '''

    :param qus_df: question data_frame
    :param ans_df: answer data_frame
    :param user_df: user data_frame
    :return: train and test
    '''

    final_df = ans_df.merge(user_df, left_on="OwnerUserId", right_on="id")
    final_df = final_df.merge(qus_df, left_on="ParentId", right_on="Id", suffixes=("_Ans", "_Qus"), how="left")

    final_df.IsAcceptedAnswer = final_df.IsAcceptedAnswer.astype(int)
    final_df.dropna(inplace=True)

    train, test = train_test_split(final_df, test_size=0.25)
    return train, test


def load_gensim_models(file_name):
    pass


# TODO - move to nlp module
def apply_gensim_model_on_df(gensim_model, df, column='clean_body', is_ans=False):
    '''

    :param gensim_model: gensim model generated for column
    :param df:
    :param column:
    :return: data_frame with more columns, each new column assigned to topic, the value is how much the row belongs to
    that topic
    '''
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
    bows = []
    for i, row in tqdm(df.iterrows()):
        #if i == 100:
        #    break
        if type(row[column]) != float:
            bow = dictionary.doc2bow(row[column].split(" "))
            t = gensim_model.get_document_topics(bow)
            for j in t:
                df.loc[i, "topic_"+str(j[0])+"_"+suffix] = j[1]
    return df

# TODO - choose what set of features we want for analysis and document
def build_classfier_and_predict(train_df, test_df):
    # all features
    features = ['Score_Ans', 'topic_0_qus_Ans', 'topic_1_qus_Ans', 'topic_2_qus_Ans', 'topic_3_qus_Ans',
                'topic_4_qus_Ans', 'topic_5_qus_Ans', 'topic_6_qus_Ans', 'topic_7_qus_Ans', 'topic_8_qus_Ans',
                'topic_9_qus_Ans', 'answer_total_score', 'closeness_centrality', 'in_degree', 'neg_answers',
                'neg_questions', 'out_degree', 'Score_Qus', 'topic_0_qus_Qus', 'topic_1_qus_Qus',
                'topic_2_qus_Qus', 'topic_3_qus_Qus', 'topic_4_qus_Qus', 'topic_5_qus_Qus', 'topic_6_qus_Qus',
                'topic_7_qus_Qus', 'topic_8_qus_Qus', 'topic_9_qus_Qus']

    # only nlp
    features = ['topic_0_qus_Ans', 'topic_1_qus_Ans', 'topic_2_qus_Ans', 'topic_3_qus_Ans',
                'topic_4_qus_Ans', 'topic_5_qus_Ans', 'topic_6_qus_Ans', 'topic_7_qus_Ans', 'topic_8_qus_Ans',
                'topic_9_qus_Ans', 'topic_0_qus_Qus', 'topic_1_qus_Qus',
                'topic_2_qus_Qus', 'topic_3_qus_Qus', 'topic_4_qus_Qus', 'topic_5_qus_Qus', 'topic_6_qus_Qus',
                'topic_7_qus_Qus', 'topic_8_qus_Qus', 'topic_9_qus_Qus']

    # nlp and features from user + graph
    features = ['topic_0_qus_Ans', 'topic_1_qus_Ans', 'topic_2_qus_Ans', 'topic_3_qus_Ans',
                'topic_4_qus_Ans', 'topic_5_qus_Ans', 'topic_6_qus_Ans', 'topic_7_qus_Ans', 'topic_8_qus_Ans',
                'topic_9_qus_Ans', 'topic_0_qus_Qus', 'topic_1_qus_Qus',
                'topic_2_qus_Qus', 'topic_3_qus_Qus', 'topic_4_qus_Qus', 'topic_5_qus_Qus', 'topic_6_qus_Qus',
                'topic_7_qus_Qus', 'topic_8_qus_Qus', 'topic_9_qus_Qus', 'in_degree', 'closeness_centrality',
                'out_degree', 'neg_questions', 'neg_answers', 'Score_Ans', 'Score_Qus']

    target_feature = 'IsAcceptedAnswer'

    # logistic regression
    logisticRegr = LogisticRegression(max_iter=1000, class_weight='balanced')
    logisticRegr.fit(train_df[features], train_df[target_feature])

    pred = logisticRegr.predict(test_df[features])
    real = test_df[target_feature]

    score = logisticRegr.score(test_df[features], test_df[target_feature])
    print "Logistic regression score is: " + str(score)
    print logisticRegr.coef_

    # random forest
    clf = RandomForestClassifier(n_estimators = 100, max_features = 6, max_depth = 6, min_samples_split=5,
                                 min_samples_leaf=5, class_weight='balanced')
    clf.fit(train_df[features], train_df[target_feature])
    score = clf.score(test_df[features], test_df[target_feature])
    print "random forest score is: " + str(score)
    print clf.feature_importances_


    #logistic regression only Ans Score
    logisticRegr = LogisticRegression(max_iter=100, class_weight='balanced')
    logisticRegr.fit(train_df[['Score_Ans', 'Score_Qus']], train_df[target_feature])

    pred = logisticRegr.predict(test_df[['Score_Ans', 'Score_Qus']])
    real = test_df[target_feature]

    score = logisticRegr.score(test_df[['Score_Ans', 'Score_Qus']], test_df[target_feature])
    print "Logistic regression only Ans score is: " + str(score)
    print logisticRegr.coef_


    # SVM classfier
    print "running SVM"
    clf = SVC(class_weight='balanced')
    clf.fit(train_df[features], train_df[target_feature])
    pred = clf.predict(test_df[features])
    print "SVM score is: " + str(1 - len(np.nonzero(pred - real)[0]) / float(len(real)))

    print "-----------------"

    print len(real)
    print pred
    print len(np.nonzero(pred - real)[0])
    print len(np.nonzero(pred - real)[0]) / len(real)

    return len(np.nonzero(pred - real)[0]) / float(len(real))


if __name__ == "__main__":

    #ans_df = pd.read_csv('C:\\Users\\Gal\\PycharmProjects\\GroundInstances\\my_tmp_2.csv')
    #qus_df = pd.read_csv('C:\\Users\\Gal\\PycharmProjects\\GroundInstances\\my_tmp.csv')
    #user_df = pd.read_csv('C:\\Users\\Gal\\Documents\\Library-Science\\user_dataframe.csv')

    # lda = gensim.models.ldamodel.LdaModel
    # model = lda.load('C:\\Users\\Gal\\Documents\\Library-Science\\my_lda_model_ans_10_topics')

    # new_df = apply_gensim_model_on_df(model, ans_df)
    # new_df.to_csv("my_tmp_2.csv")

    #train, test = combine_qus_and_ans(qus_df, ans_df, user_df)
    #train.to_csv("C:\\Users\\Gal\\Documents\\Library-Science\\combined_train.csv")
    #test.to_csv("C:\\Users\\Gal\\Documents\\Library-Science\\combined_test.csv")

    train = pd.read_csv("C:\\Users\\Gal\\Documents\\Library-Science\\combined_train.csv")
    print "Done read train"
    test = pd.read_csv("C:\\Users\\Gal\\Documents\\Library-Science\\combined_test.csv")
    print "Done read test"

    print build_classfier_and_predict(train, test)

    print "Done"

    #q_docs = qus_df['clean_body']
    #q_docs = [x.split(" ") for x in q_docs if type(x) != float]
    #q_dictionary = corpora.Dictionary(q_docs)
    #corpora = model.id2word

    #print "done"

    #combine_qus_and_ans()

