import pandas as pd
from tqdm import tqdm
from config import PROJECT_DIR


def add_cluster_0_1_to_question_by_Tag_sets(tags_clusters, df):
    # ADD foreach cluster i the field cluster_i:
    #   cluster_i[question] = 0/1 if the question's
    #   tags not/appear in the cluster

    cluster_2_tags = {}

    for cluster, gr in tags_clusters.groupby('cluster'):
        if cluster != 'None':
            cluster_name = 'cluster_' + str(cluster)
            cluster_2_tags[cluster_name] = set(gr['Tag']) # there is just one cluster - so it's OK!

    num_clusters = len(cluster_2_tags.keys())

    # create new columns
    for i in range(num_clusters):
        df['cluster_' + str(i)] = 0

    for i in range(num_clusters):
        cluster_name = 'cluster_' + str(i)
        tmp_series = df['all_tags'].apply(lambda tag_set: (set(tag_set) & set(cluster_2_tags[cluster_name])) != set())
        df[cluster_name] = tmp_series.astype(int)

    return df


def add_users_scores_clusters_to_train(df):
    # WARNING: this function consider the result of isAcceptedAnswer
    # ONLY USE ON TRAIN
    #
    # ADD fields:
    #   user_socre_cluster_i[u_id] =
    #       (#accepted_answers / #total_answers) that user u_id did on cluster i
    #       0 if user u_id didn't answer any ques of cluster i
    #   user_score_total[u_id] =
    #       (#accepted_answers / #total_answers) that user u_id did
    #       0 if user u_id didn't answer any ques

    u_id = 'OwnerUserId_ans'
    is_acc = 'IsAcceptedAnswer'
    all_clstrs = filter(lambda field: str(field).startswith('cluster_'), list(df))

    # score by clusters:
    for cluster in all_clstrs:
        # D[user_id] = [#answers, #true_answers]
        D = {}
        for user, gr in tqdm(df.groupby(u_id)):
            D[user] = [0, 0]
            D[user][0] = gr[cluster].sum()
            D[user][1] = (gr.loc[gr[is_acc] == True][cluster]).sum()
        tmp_series = df[u_id].apply(lambda user: (0 if D[user][0] == 0 else D[user][1] / float(D[user][0])))
        df['user_score_' + cluster] = tmp_series.astype(float)

    # general_score = user_amount_true / user_amount_questions
    # D[user_id] = [#answers, #true_answers, diff_clusters]
    D = {}
    for user, gr in tqdm(df.groupby(u_id)):
        D[user] = [0, 0]
        D[user][0] = len(gr)
        D[user][1] = len(gr.loc[gr[is_acc] == True])
    tmp_series = df[u_id].apply(lambda user: (0 if D[user][0] == 0 else D[user][1] / float(D[user][0])))
    df['user_score_general'] = tmp_series.astype(float)

    return df


def add_users_scores_clusters_to_test(df, train):
    # foreach u_id, joins it's user scores,
    # as pre-calculated in the train.
    # if user doesn't appear in train => gets zero

    u_id = 'OwnerUserId_ans'
    fields_for_join = filter(lambda field: str(field).startswith('user_score_'), list(train))
    fields_for_join = [u_id] + fields_for_join
    df2 = train[fields_for_join].groupby(u_id)[fields_for_join].first()

    df = df.merge(df2, left_on=u_id, right_on=u_id, how='left')
    df = df.fillna(0)
    return df


def create_user_total_scores_by_clusters(df):
    # calc total_score_user_question_by_clusters
    # assuming having the users score per question+cluster
    # as was calculated in the training set.
    # adding:
    # total_score_user_question_by_clusters = <(q_clusters),(users_scores_by_cluster)>
    # total_score_user_question_by_clusters_relative = <(q_clusters),(users_scores_by_cluster)> / #clusters_of_q

    all_clstrs = filter(lambda field: str(field).startswith('cluster_'), list(df))

    amount_tags_per_q = df[all_clstrs].sum(1)
    for_inner_multi = pd.DataFrame()

    for cluster in all_clstrs:
        for_inner_multi[cluster] = df[cluster] * df['user_score_' + cluster]
    s = for_inner_multi[all_clstrs].sum(1)

    df['total_score_user_question_by_clusters'] = s.astype(float)
    df['total_score_user_question_by_clusters_relative'] = (s.astype(float) / amount_tags_per_q.astype(float)).fillna(0)

    return df


def add_list_of_tags_to_question(df, tags):
    # ADD field 'all_tags'
    # that is a list of all the tags of the question

    tags = tags.groupby('Id')['Tag'].apply(list).reset_index(name='all_tags')
    df['index_for_join'] = df['ParentId']
    df = df.set_index('index_for_join').join(tags.set_index('Id'))
    df.loc[df['all_tags'].isnull(), ['all_tags']] = df.loc[df['all_tags'].isnull(), 'all_tags'].apply(lambda x: [])
    return df


def tags_to_questions(tags, df_train, df_test, tags_clusters):
    # train = large(first)
    # test = small(second)

    u_id = 'OwnerUserId_ans'
    df_test = df_test.fillna(0)
    df_train = df_train.fillna(0)

    # create train:
    df_train[u_id] = df_train[u_id].fillna(0)
    df_train = add_list_of_tags_to_question(df_train, tags)
    df_train = add_cluster_0_1_to_question_by_Tag_sets(tags_clusters, df_train)
    df_train = add_users_scores_clusters_to_train(df_train)
    df_train = create_user_total_scores_by_clusters(df_train)
    df_train.to_csv(PROJECT_DIR + '/train_with_tag_clusters_and_user_scores.csv', index=False)

    # create test:
    df_test[u_id] = df_test[u_id].fillna(0)
    df_test = add_list_of_tags_to_question(df_test, tags)
    df_test = add_cluster_0_1_to_question_by_Tag_sets(tags_clusters, df_test)
    df_test = add_users_scores_clusters_to_test(df_test, df_train)
    df_test = create_user_total_scores_by_clusters(df_test)
    df_test.to_csv(PROJECT_DIR + '/test_with_tag_clusters_and_user_scores.csv', index=False)

