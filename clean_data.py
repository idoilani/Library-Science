import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
import nlp_features
import create_user_profile

PROJECT_DIR = r'C:\Users\Gal\Documents\Library-Science'


def clean_data(file_obj, final_file_name):
    """
    Reads data frame and saves it back without the nulls
    """
    data_frame = pd.read_csv(file_obj)
    data_frame.dropna(axis=0, inplace=True)
    data_frame.to_csv(final_file_name)
    return


def split_to_train_test(qus_df, ans_df, tag_df, train_fraq=0.7):
    '''
    split to train and test and add features related to user
    :param qus_df:
    :param train_fraq:
    :return: same data frame with a new column
    '''

    # split to train and test
    qus_df = shuffle(qus_df)
    n_train = int(len(qus_df) * train_fraq)
    qus_df_train = qus_df[:n_train]
    qus_df_test = qus_df[n_train + 1:]

    # creates clean body in qus for NLP features
    D_qus = nlp_features.count_words_in_dataframe(qus_df_train, 'Body')
    qus_df_train = nlp_features.remove_irrelevant_words(qus_df_train, D_qus, 'Body', 'clean_qus')
    qus_df_test = nlp_features.remove_irrelevant_words(qus_df_test, D_qus, 'Body', 'clean_qus')

    merged_df_train = ans_df.merge(qus_df_train, left_on="ParentId", right_on='Id', suffixes=("_ans", "_qus"),
                                   how="inner")

    # creates clean body column in ans for NLP features
    D_ans = nlp_features.count_words_in_dataframe(merged_df_train, 'Body_ans')
    merged_df_train = nlp_features.remove_irrelevant_words(merged_df_train, D_ans, 'Body_ans', 'clean_ans')

    # build user_df from train
    users = create_user_profile.list_of_users(merged_df_train)
    users.create_user_graph()
    users.set_node_properties()

    # creates and save data frame for users based on train data frame
    users.set_user_scores(merged_df_train)
    user_df = users.generate_data_frame(PROJECT_DIR + '\data\user_dataframe.csv')

    # build tags dict
    tag_dict = defaultdict(set)
    for Id, gr in tag_df.groupby("Id"):
        tag_dict[Id] = set(gr['Tag'])

    # add users features to train
    merged_df_train['Tag_sets'] = merged_df_train['Id_qus'].apply(lambda qus_id: tag_dict[qus_id])
    merged_df_train = merged_df_train.merge(user_df, left_on="OwnerUserId_ans", right_on="id_user", how="left")
    merged_df_test = ans_df.merge(qus_df_test, left_on="ParentId", right_on='Id', suffixes=("_ans", "_qus"),
                                  how="inner")
    # add users features to test
    merged_df_test = nlp_features.remove_irrelevant_words(merged_df_test, D_ans, 'Body_ans', 'clean_ans')
    merged_df_test['Tag_sets'] = merged_df_test['Id_qus'].apply(lambda qus_id: tag_dict[qus_id])
    merged_df_test = merged_df_test.merge(user_df, left_on="OwnerUserId_ans", right_on="id_user", how="left")

    print "saving data frames"
    merged_df_train.to_csv(PROJECT_DIR + '\data\\train_df.csv')
    merged_df_test.to_csv(PROJECT_DIR + '\data\\test_df.csv')

    return merged_df_train, merged_df_test



