# imports
import pandas as pd
import numpy as np
import networkx as nx
import sys
from create_user_profile import *
import clean_data
import nlp_features
import gensim
import add_time_features
import sapir_features
import user_reputation

# TODO - add documentation on this module
# TODO - start a documentation of the whole project (in a doc)

# TODO - move to configuration file
# load data
PROJECT_DIR = r'C:\Users\Gal\Documents\Library-Science'
CLEAN_DATA = True
CREATE_USER_GRAPH = False
LOAD_LDA_MODEL = True
LOAD_MODEL_WITH_NLP_FEATURES = True
ADD_SAPIR_FEATURES = True
ADD_USER_REPUTATION = True
# ans = pd.read_csv(project_dir + '/Answers.csv')
# qus = pd.read_csv(project_dir + '/Questions.csv')
# tags = pd.read_csv(project_dir + '/Tags.csv')
if ADD_USER_REPUTATION:
    train_df = pd.read_csv(r'C:/Users/Gal/Documents/Library-Science/data/train_with_tags.csv')
    test_df = pd.read_csv(r'C:/Users/Gal/Documents/Library-Science/data/test_with_tags.csv')

    # creates new features for train
    print "add user reputation"
    train_df = user_reputation.users_scores_clusters(train_df)
    train_df = user_reputation.create_user_total_scores_by_clusters(train_df)
    train_df.to_csv(r'C:/Users/Gal/Documents/Library-Science/data/train_with_user_profile.csv')

    # create new features for test
    exit()

if ADD_SAPIR_FEATURES:
    # train_df = pd.read_csv(PROJECT_DIR + '\\data\\train_with_time_diff.csv')
    # test_df = pd.read_csv(PROJECT_DIR + '\\data\\test_with_time_diff.csv')

    train_df = pd.read_csv(PROJECT_DIR + '\\data\\train_with_hir.csv')
    test_df = pd.read_csv(PROJECT_DIR + '\\data\\test_with_hir.csv')

    tags_df = pd.read_csv(PROJECT_DIR + '\\data\\tags_clusters2.csv')

    train_df = sapir_features.tags_to_questions(tags_df, train_df)
    test_df = sapir_features.tags_to_questions(tags_df, test_df)

    train_df.to_csv(r'C:/Users/Gal/Documents/Library-Science/data/train_with_tags.csv')
    test_df.to_csv(r'C:/Users/Gal/Documents/Library-Science/data/test_with_tags.csv')

    exit()

    train_df = sapir_features.insert_hirerchy_answer_col(train_df)
    test_df = sapir_features.insert_hirerchy_answer_col(test_df)

    train_df.to_csv(r'C:/Users/Gal/Documents/Library-Science/data/train_with_hir.csv')
    test_df.to_csv(r'C:/Users/Gal/Documents/Library-Science/data/test_with_hir.csv')

    exit()




if LOAD_MODEL_WITH_NLP_FEATURES:
    train_df = pd.read_csv(PROJECT_DIR + '\\data\\train_df_with_nlp_topics.csv')
    test_df = pd.read_csv(PROJECT_DIR + '\\data\\test_df_with_nlp_topics.csv')

    add_time_features.add_time_diff_features(train_df, True)
    add_time_features.add_time_diff_features(test_df, False)
    exit()


# clean the data
if CLEAN_DATA:
    print "cleaning data"
    clean_data.clean_data(PROJECT_DIR + '\data\Answers.csv', PROJECT_DIR + '\data\Clean_Answers.csv')
    clean_data.clean_data(PROJECT_DIR + '\data\Questions.csv', PROJECT_DIR + '\data\Clean_Questions.csv')
    clean_data.clean_data(PROJECT_DIR + '\data\Tags.csv', PROJECT_DIR + '\data\Clean_Tags.csv')

    print "reading data"
    ans = pd.read_csv(PROJECT_DIR + '\data\Clean_Answers.csv')
    qus = pd.read_csv(PROJECT_DIR + '\data\Clean_Questions.csv')
    tags = pd.read_csv(PROJECT_DIR + '\data\Clean_Tags.csv')

# USER PROFILE SECTION
# TODO - maybe add a special feature for not very popular users - uncomment it
# build a profile for each user

if CREATE_USER_GRAPH:
    print "creating users graph"
    users = list_of_users(qus, ans)
    users.create_user_graph()
    users.set_node_properties()
    users.set_user_scores(ans, qus)


# TODO - write a function setting the relevant features for test table - maybe we split our data to 2 tables

if CLEAN_DATA:
    train_df, test_df = clean_data.split_to_train_test(qus, ans, tags, create_user_graph=CREATE_USER_GRAPH)

else:
    print "loading data..."
    train_df = pd.read_csv(r'C:\Users\Gal\Documents\Library-Science\data\train_df.csv')
    test_df = pd.read_csv(r'C:\Users\Gal\Documents\Library-Science\data\test_df.csv')


print "start nlp"
# NLP SECTION
# First - cleaning questions and answers for train and test
# TODO - get train_df, and test_df from illani and gidi

# DONE in split to train and test

# print "removing irrelevant words"
# D = nlp_features.count_words_in_dataframe(train_df, 'clean_body_Qus')
# train_df = nlp_features.remove_irrelevant_words(train_df, D)

# D = nlp_features.count_words_in_dataframe(train_df, 'clean_body_Ans')
# train_df = nlp_features.remove_irrelevant_words(train_df, D)

# D = nlp_features.count_words_in_dataframe(test_df, 'clean_body_Qus')
# test_df = nlp_features.remove_irrelevant_words(test_df, D)

# D = nlp_features.count_words_in_dataframe(test_df, 'clean_body_Ans')
# test_df = nlp_features.remove_irrelevant_words(test_df, D)

print "building lda models"
# building models for qus and ans

if not LOAD_LDA_MODEL:
    lda_qus = nlp_features.my_lda(train_df, 'clean_qus', number_of_topics=10, lda_model_name="my_lda_model_qus_10_topics")
    lda_ans = nlp_features.my_lda(train_df, 'clean_ans', number_of_topics=10, lda_model_name="my_lda_model_ans_10_topics")

else:
    lda = gensim.models.ldamodel.LdaModel
    lda_qus = lda.load("C:\\Users\\Gal\\Documents\\Library-Science\\my_lda_model_qus_10_topics")
    lda_ans = lda.load("C:\\Users\\Gal\\Documents\\Library-Science\\my_lda_model_ans_10_topics")

print "Done building the models"
# setting new features
print "applying gensim models on train and test data frames"
train_df = nlp_features.apply_gensim_model_on_df(lda_qus, train_df, "clean_qus", False)
train_df = nlp_features.apply_gensim_model_on_df(lda_ans, train_df, "clean_ans", True)

test_df = nlp_features.apply_gensim_model_on_df(lda_qus, test_df, "clean_qus", False)
test_df = nlp_features.apply_gensim_model_on_df(lda_ans, test_df, "clean_ans", True)

train_df.to_csv(PROJECT_DIR + '\\data\\train_df_with_nlp_topics.csv')
test_df.to_csv(PROJECT_DIR + '\\data\\test_df_with_nlp_topics.csv')

if __name__ == "__main__":
    # load data
    print "Done"