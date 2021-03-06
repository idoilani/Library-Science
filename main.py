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
import tag_clusters
import answers_hierarchy
import tag_clusters_insert_features
from config import PROJECT_DIR
# load data
LOAD_LDA_MODEL = False


# clean the data and add features related to users
def main():
    print "cleaning data and save it back to the same directory"
    clean_data.clean_data(PROJECT_DIR + '\data\Answers.csv', PROJECT_DIR + '\data\Clean_Answers.csv')
    clean_data.clean_data(PROJECT_DIR + '\data\Questions.csv', PROJECT_DIR + '\data\Clean_Questions.csv')
    clean_data.clean_data(PROJECT_DIR + '\data\Tags.csv', PROJECT_DIR + '\data\Clean_Tags.csv')

    print "reading it back..."
    ans = pd.read_csv(PROJECT_DIR + '\data\Clean_Answers.csv')
    qus = pd.read_csv(PROJECT_DIR + '\data\Clean_Questions.csv')
    tags = pd.read_csv(PROJECT_DIR + '\data\Clean_Tags.csv')

    print "Split to train and test and prepare user data_frame from train and add columns for the NLP features"
    train_df, test_df = clean_data.split_to_train_test(qus, ans, tags)

    # build LDA models for answers and questions
    if not LOAD_LDA_MODEL:
        lda_qus = nlp_features.my_lda(train_df, 'clean_qus', number_of_topics=10, lda_model_name="my_lda_model_qus_10_topics")
        lda_ans = nlp_features.my_lda(train_df, 'clean_ans', number_of_topics=10, lda_model_name="my_lda_model_ans_10_topics")

    # load LDA models
    else:
        lda = gensim.models.ldamodel.LdaModel
        lda_qus = lda.load(PROJECT_DIR + "\\my_lda_model_qus_10_topics")
        lda_ans = lda.load(PROJECT_DIR + "\\my_lda_model_ans_10_topics")

    print "applying gensim models on train and test data frames"
    train_df = nlp_features.apply_gensim_model_on_df(lda_qus, train_df, "clean_qus", False)
    train_df = nlp_features.apply_gensim_model_on_df(lda_ans, train_df, "clean_ans", True)

    test_df = nlp_features.apply_gensim_model_on_df(lda_qus, test_df, "clean_qus", False)
    test_df = nlp_features.apply_gensim_model_on_df(lda_ans, test_df, "clean_ans", True)

    # add features related to time hierarchy and save a temporary data frame
    train_df = add_time_features.add_time_diff_features(train_df, True)
    test_df = add_time_features.add_time_diff_features(test_df, False)

    # save date_frames with nlp features
    train_df.to_csv(PROJECT_DIR + '\\data\\train_df_with_nlp_topics.csv', index=False)
    test_df.to_csv(PROJECT_DIR + '\\data\\test_df_with_nlp_topics.csv', index=False)

    # add answer hierarchy
    train_df = answers_hierarchy.insert_hirerchy_answer_col(train_df)
    test_df = answers_hierarchy.insert_hirerchy_answer_col(test_df)

    # create tags clusters and add related features
    tag_clusters.create_tag_clusters(tags, "tag_clusters.csv")
    tags_cluster_df = pd.read_csv('tag_clusters.csv')
    tag_clusters_insert_features.tags_to_questions(tags, train_df, test_df, tags_cluster_df)

    # save final data frame
    train_df.to_csv(PROJECT_DIR + "/data/train_with_user_profile.csv", index=False)
    test_df.to_csv(PROJECT_DIR + "/data/test_with_user_profile.csv", index=False)


if __name__ == "__main__":
    main()
    print "Done"