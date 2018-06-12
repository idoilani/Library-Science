import pandas as pd
from tqdm import tqdm

def tags_to_questions(tags_clusters, df):
    cluster_2_tags = {}

    for cluster, gr in tags_clusters.groupby('cluster'):
        if cluster != 'None':
            print cluster
            cluster_name = 'cluster_' + str(cluster)
            cluster_2_tags[cluster_name] = set(gr['Tag'])

    num_clusters = len(cluster_2_tags.keys())

    # create new columns
    for i in range(num_clusters):
        df['cluster_' + str(i)] = 0

    print "Done creating new columns"

    for i in tqdm(range(num_clusters)):
        cluster_name = 'cluster_' + str(i)
        tmp_series = df['Tag_sets'].apply(lambda tag_set: (eval(tag_set) & cluster_2_tags[cluster_name]) != set())
        tmp_series = map(int, tmp_series)
        df[cluster_name] = tmp_series

    return df


def insert_hirerchy_answer_col(df):
    # inserts hirerchy field:
    # the number of the answer by chronological order
    date = 'CreationDate_ans'
    pid = 'ParentId' #it's the question id
    hir = 'hirerchy'

    # sort values by times
    df = df.sort_values(by=[pid, date])

    qus_id_2_hir = {}
    for pid, gr in df.groupby(pid):
        Id_qus = gr['Id_qus']
        for i, Id in enumerate(Id_qus):
            qus_id_2_hir[Id] = i + 1

    df[hir] = df['Id_qus'].apply(lambda q_id:qus_id_2_hir[q_id])

    return df


def add_set_of_tags_to_question(df, tags):
    tags = tags.groupby('Id')['Tag'].apply(set).reset_index(name='tags')
    df = df.set_index('ParentId').join(tags.set_index('Id'))
    return df