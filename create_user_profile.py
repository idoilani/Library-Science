import collections
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt

class User():
    def __init__(self, user_id):
        self.id = user_id
        self.question_total_score = 0
        self.answer_total_score = 0
        self.number_of_answers = 0

        self.neg_questions = 0
        self.neg_answers = 0

        # Ids of other posts
        self.question_ids = set()
        self.answers_ids = set()

        # list of related users
        self.users_who_replied_to_question = []
        self.users_who_was_answered_by_this_user = collections.defaultdict()

        # Nodes properties
        self.closeness_centrality = -1
        self.in_degree = 0
        self.out_degree = 0

    def process_answer_df(self, df):
        self.users_who_replied_to_question = list(set(df['OwnerUserId']))
        self.number_of_answers = len(df['OwnerUserId'])
        self.answers_ids = set(df['Id'])

    def process_question_df(self, df):
        self.question_ids = set(df["Id"])

    def set_question_total_score(self, qus):
        self.question_total_score = sum(qus[qus["Id"].isin(self.question_ids)]['Score'])

    def set_number_of_negative_questions(self, qus):
        self.neg_questions = len([x for x in qus[qus["Id"].isin(self.question_ids)]['Score'] if x < 0])

    def set_number_of_negative_answers(self, ans):
        self.neg_answers = len([x for x in ans[ans["Id"].isin(self.answers_ids)]['Score'] if x < 0])

    def set_answer_total_score(self, ans):
        self.answer_total_score = sum(ans[ans["Id"].isin(self.answers_ids)]['Score'])

    def set_closeness_centrality(self, closeness_centrality):
        self.closeness_centrality = closeness_centrality

    def set_in_degree(self, in_degree):
        self.in_degree = in_degree

    def set_out_degree(self, out_degree):
        self.out_degree = out_degree

class list_of_users():
    def __init__(self, qst, ans):
        self.graph = nx.DiGraph()
        self.users = []

        for user_id, gr in qst.groupby(["OwnerUserId"]):
            new_user = User(user_id)
            new_user.process_question_df(gr)
            self.users.append(new_user)

        for user in self.users:
            tmp_df = ans[ans['ParentId'].isin(user.question_ids)]
            user.process_answer_df(tmp_df)

    def create_user_graph(self):
        '''
        :param users: a list of users (
        :return: networkx objects represents the connections between users
        '''
        for user in self.users:
            for Id in user.users_who_replied_to_question:
                self.graph.add_edge(Id, user.id)

    def save_obj(self, output):
        pickle.dump(self, file(output, 'wb'), pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(output):
        return pickle.load(file(output, 'rb'))

    def set_node_properties(self):
        sub_graphs = list(nx.weakly_connected_component_subgraphs(self.graph))
        big_sub_graphs = [x for x in sub_graphs if len(x) > 20]
        if len(big_sub_graphs) > 1:
            print "There is more than 1 big sub-graph"

        print "before"
        D_bet = nx.closeness_centrality(self.graph)
        D_in = nx.in_degree_centrality(self.graph)
        D_out = nx.out_degree_centrality(self.graph)
        print "after"

        for user in self.users:
            if user.id in D_bet.keys():
                user.set_closeness_centrality(D_bet[user.id])
                user.set_in_degree(D_in[user.id])
                user.set_out_degree(D_out[user.id])

    def set_user_scores(self, ans, qus):
        for user in self.users:
            user.set_answer_total_score(ans)
            user.set_question_total_score(qus)
            user.set_number_of_negative_questions(qus)
            user.set_number_of_negative_answers(ans)

    def generate_data_frame(self, output_file=None):
        features = ["id", "question_total_score", "answer_total_score", "number_of_answers", "neg_questions",
                    "neg_answers", "closeness_centrality", "in_degree", "out_degree"]
        df = pd.DataFrame([{val : getattr(x, val) for val in features} for x in self.users])
        if output_file == None:
            return df
        else:
            df.to_csv(output_file, index=False)
            return df



if __name__ == "__main__":
    """
    a = list_of_users.load_obj("C:\Users\Gal\Documents\Library-Science\user_list.pkl")
    print len(a.graph)
   # nx.draw(a.graph)
   # plt.show()
    print [(a.users[x].closeness_centrality, a.users[x].in_degree, a.users[x].out_degree) for x in range(20) if a.users[x].out_degree != 0]
    print [(a.users[x].answer_total_score, a.users[x].question_total_score) for x in range(20)]
    print [(a.users[x].neg_answers, a.users[x].neg_questions) for x in range(20)]
    """


    project_dir = '/Users/Gal/Downloads/rquestions'
    ans = pd.read_csv(project_dir + '/Answers.csv')
    qus = pd.read_csv(project_dir + '/Questions.csv')
    tags = pd.read_csv(project_dir + '/Tags.csv')

    users = list_of_users(qus, ans)
    users.create_user_graph()
    users.set_node_properties()
    users.set_user_scores(ans, qus)
    df = users.generate_data_frame("C:\Users\Gal\Documents\Library-Science\user_dataframe.csv")
    users.save_obj("C:\Users\Gal\Documents\Library-Science\user_list.pkl")







