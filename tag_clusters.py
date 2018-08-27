import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import collections
from sklearn.cluster import SpectralClustering
import itertools
from numpy import linalg as la


def create_dict_from_dataframe(data_tags, field_key, field_value):
    # D[field_key] = [val1, val2, ..., valn]
    # vals are the dieffrent values of field_value appeared with field_key
    #D = {tag: tag_info["Id"].tolist() for tag, tag_info in data_tags.groupby("Tag")}
    D = {key: key_info[field_value].tolist() for key, key_info in data_tags.groupby(field_key)}
    return D


def create_graph_from_dict(D):
    G = nx.Graph()
    i = 0
    for value in D.itervalues():
        for cpl in itertools.combinations(value, r=2):
            q1, q2 = cpl[0], cpl[1]
            if G.has_edge(q1, q2):
                G[q1][q2]['weight'] += 1
            else:
                G.add_edge(q1, q2, weight=1)
            i += 1
            if(i % 1000000 == 0):
                print i
    return G


def save_graph(G):
    return


def draw_weighted_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos)
    plt.show()


def create_clusters_from_graph(G, amount_clusters=50, amount_iterations=10000):
    adj_mat = nx.to_numpy_matrix(G)
    sc = SpectralClustering(amount_clusters, affinity='precomputed', n_init=amount_iterations)
    sc.fit(adj_mat)
    return sc


def save_clusters(G, sc, data_tags, name_file):
    all_nodes = list(G.nodes)
    reslts = {'Tag': all_nodes, 'cluster': sc.labels_,
              'amount_ques_tag_in': [len(data_tags.loc[data_tags['Tag'] == i]) for i in all_nodes]}
    df = pd.DataFrame(data=reslts)
    tags_not_included = list(set(data_tags['Tag']) - set(df['Tag']))
    tags_not_in_graph = {'Tag': tags_not_included, 'cluster': ['None' for i in xrange(len(tags_not_included))],
                         'amount_ques_tag_in': [len(data_tags.loc[data_tags['Tag'] == i]) for i in tags_not_included]}
    df2 = pd.DataFrame(data=tags_not_in_graph)
    df = df.append(df2)
    df.to_csv(name_file)


def print_all_clusters(G, sc):
    print('----------\nspectral clustering:')
    array_nodes = [i for i in G.nodes]
    results = [[array_nodes[i], sc.labels_[i]] for i in xrange(0, len(G.nodes))]
    results = sorted(results, key=lambda x: x[1])
    res_by_labels = [(clstr, [k[0] for k in filter(lambda p: p[1] == clstr, results)]) for clstr in
                     set([j[1] for j in results])]
    print "\nclusters sizes:", map(lambda x: len(x[1]), res_by_labels)
    for i in res_by_labels:
        print len(i[1]),"pieces in cluster:", i[0]
        if len(i[1]) <= 200:
            print i
        else:
            print "over 200 items"


def save_graph(G, file_name):
    nx.write_gpickle(G, file_name)


def upload_graph(file_name):
    G = nx.read_gpickle(file_name)
    return G


def print_dictionary_statistics(D):
    print "#keys in D:", len(D)
    print "#keys with len(val)>1:", len(filter(lambda x: len(x)>1, D.itervalues()))
    amount_qs_per_tag = [len(i) for i in D.itervalues()]
    counter = collections.Counter(amount_qs_per_tag)
    print "len(val): #keys ->", counter
    print "keys with len(val)>100:", sorted([(i, len(D[i])) for i in filter(lambda x: len(D[x])>100, D.iterkeys())],key=(lambda x: x[1]))

    print "#edges:", sum([int(i*(i-1)*0.5*counter[i]) for i in counter.iterkeys()])


def print_graph_statistics(G):
    print "---------\nGraph statistics:"
    print "#nodes in G:", len(G)
    print "#conn components in G:", int(nx.number_connected_components(G))
    print "sizes of conn components:", map(lambda x: len(x), list(nx.connected_component_subgraphs(G)))
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)

    print sum([degree_count[i] for i in xrange(21)])
    weight_seq =  map(lambda x: x['weight'], G.edges.values())
    weight_count = collections.Counter(weight_seq)
    print "graph degrees:", degree_count
    print "graph weights:", weight_count
    print "edges in G:", len(G.edges)


def add_edges_between_2distance_nodes_to_G(G):
    adj_mat = nx.to_numpy_matrix(G)
    adj_mat_dbld = la.matrix_power(adj_mat, 2)
    for node1_ind in xrange(len(G)):
        for node2_ind in xrange(node1_ind):
            if adj_mat_dbld.item(node1_ind, node2_ind) >= 1:
                node1 = list(G.nodes)[node1_ind]
                node2 = list(G.nodes)[node2_ind]
                if G.has_edge(node1, node2):
                    G[node1][node2]['weight'] += 1
                else:
                    G.add_edge(node1, node2, weight=1)
    return G


def create_and_save_tags_graph(tags, name_save_graph):
    # D = {'q1': [tag1, tag2], 'q2': [tag1, tag3], ...}
    # G = (V=tags, E={(tag1,tag2) | tag1 and tag2 has same question)} )

    D = create_dict_from_dataframe(tags, "Id", "Tag")
    # print_dictionary_statistics(D)
    G = create_graph_from_dict(D)
    G = add_edges_between_2distance_nodes_to_G(G)
    save_graph(G, name_save_graph)


def create_tag_clusters(tags, name_tag_clusters_table):
    # for debugging: tags = tags.head(100)
    create_and_save_tags_graph(tags, "tags_graph.gpickle")
    G = upload_graph("tags_graph.gpickle")

    # taking the maximum componenet:
    biggest_conn = max(list(nx.connected_component_subgraphs(G)), key=lambda conn: len(conn))
    G = biggest_conn

    amount_clusters = 50
    amount_iterations = 3
    sc = create_clusters_from_graph(G, amount_iterations=amount_iterations, amount_clusters=amount_clusters)
    save_clusters(G, sc, tags, name_tag_clusters_table)

    # optional for debugging:
    # print_all_clusters(G, sc)
    # print_graph_statistics(G)
    # draw_weighted_graph(G)

