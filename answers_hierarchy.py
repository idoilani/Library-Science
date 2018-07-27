import pandas as pd

def insert_hirerchy_answer_col(df):
    # inserts hierarchy field:
    # the number of the answer by chronological order
    date = 'CreationDate_ans'
    pid = 'ParentId' #it's the question id
    hir = 'hierarchy'

    # sort values by times
    df = df.sort_values(by=[pid, date])

    qus_id_2_hir = {}
    for pid, gr in df.groupby(pid):
        Id_qus = gr['Id_qus']
        for i, Id in enumerate(Id_qus):
            qus_id_2_hir[Id] = i + 1

    df[hir] = df['Id_qus'].apply(lambda q_id: qus_id_2_hir[q_id])
    return df
