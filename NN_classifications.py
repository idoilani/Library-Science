from keras.layers import Input, Dense, Lambda, Activation
from keras.models import Sequential
from sklearn.preprocessing import normalize
import pandas as pd

def main():
    target = 'IsAcceptedAnswer'

    train = pd.read_csv(r"C:\Users\sapir\Desktop\sadna\clean_data\train_with_tag_clusters_and_user_scores.csv")
    test = pd.read_csv(r"C:\Users\sapir\Desktop\sadna\clean_data\test_with_tag_clusters_and_user_scores.csv")
    
    feature_list = get_features()    
    num_features = len(feature_list)

    batch_size = 10
    hidden_dim1 = 50
    hidden_dim2 = 50
    epochs = 200
    print("batch_size=" ,batch_size, "\nepochs=", epochs, "\nhidden_dims=", [hidden_dim1, hidden_dim2])
    	
    #create model:
    model = Sequential([
        Dense(hidden_dim1, input_dim=num_features),
        Activation('relu'),
        Dense(hidden_dim2),
        Activation('relu'),
        Dense(1), # output layer.
        Activation('sigmoid'), # Regression via Classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train[feature_list], train[target], epochs=epochs, batch_size=batch_size, verbose=0)

    loss_and_metrics = model.evaluate(test[feature_list], test[target], batch_size=batch_size)
    print("\n---------------\nloss metrices:")
    print(loss_and_metrics)

    # classes = model.predict(test[feature_list], batch_size=batch_size)
    # print("\n---------------\nclasses:")
    # print(classes)

    scores = model.evaluate(test[feature_list], test[target])
    print("\n---------------\nscores:")
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def get_features():
    L = ['Score_ans',
        'Score_qus',
        'answer_total_score_user',
        'closeness_centrality_user',
        'in_degree_user',
        'neg_answers_user',
        'neg_questions_user',
        'number_of_answers_user',
        'out_degree_user',
        'question_total_score_user',
        'topic_0_qus',
        'topic_1_qus',
        'topic_2_qus',
        'topic_3_qus',
        'topic_4_qus',
        'topic_5_qus',
        'topic_6_qus',
        'topic_7_qus',
        'topic_8_qus',
        'topic_9_qus',
        'topic_0_ans',
        'topic_1_ans',
        'topic_2_ans',
        'topic_3_ans',
        'topic_4_ans',
        'topic_5_ans',
        'topic_6_ans',
        'topic_7_ans',
        'topic_8_ans',
        'topic_9_ans',
        'timeDiff',
        'MinResponseTime',
        'diff_percentile_bucket',
        'hirerchy',
        'cluster_0',
        'cluster_1',
        'cluster_2',
        'cluster_3',
        'cluster_4',
        'cluster_5',
        'cluster_6',
        'cluster_7',
        'cluster_8',
        'cluster_9',
        'cluster_10',
        'cluster_11',
        'cluster_12',
        'cluster_13',
        'cluster_14',
        'cluster_15',
        'cluster_16',
        'cluster_17',
        'cluster_18',
        'cluster_19',
        'cluster_20',
        'cluster_21',
        'cluster_22',
        'cluster_23',
        'cluster_24',
        'cluster_25',
        'cluster_26',
        'cluster_27',
        'cluster_28',
        'cluster_29',
        'cluster_30',
        'cluster_31',
        'cluster_32',
        'cluster_33',
        'cluster_34',
        'cluster_35',
        'cluster_36',
        'cluster_37',
        'cluster_38',
        'cluster_39',
        'cluster_40',
        'cluster_41',
        'cluster_42',
        'cluster_43',
        'cluster_44',
        'cluster_45',
        'cluster_46',
        'cluster_47',
        'cluster_48',
        'cluster_49',
        'user_score_cluster_0',
        'user_score_cluster_1',
        'user_score_cluster_2',
        'user_score_cluster_3',
        'user_score_cluster_4',
        'user_score_cluster_5',
        'user_score_cluster_6',
        'user_score_cluster_7',
        'user_score_cluster_8',
        'user_score_cluster_9',
        'user_score_cluster_10',
        'user_score_cluster_11',
        'user_score_cluster_12',
        'user_score_cluster_13',
        'user_score_cluster_14',
        'user_score_cluster_15',
        'user_score_cluster_16',
        'user_score_cluster_17',
        'user_score_cluster_18',
        'user_score_cluster_19',
        'user_score_cluster_20',
        'user_score_cluster_21',
        'user_score_cluster_22',
        'user_score_cluster_23',
        'user_score_cluster_24',
        'user_score_cluster_25',
        'user_score_cluster_26',
        'user_score_cluster_27',
        'user_score_cluster_28',
        'user_score_cluster_29',
        'user_score_cluster_30',
        'user_score_cluster_31',
        'user_score_cluster_32',
        'user_score_cluster_33',
        'user_score_cluster_34',
        'user_score_cluster_35',
        'user_score_cluster_36',
        'user_score_cluster_37',
        'user_score_cluster_38',
        'user_score_cluster_39',
        'user_score_cluster_40',
        'user_score_cluster_41',
        'user_score_cluster_42',
        'user_score_cluster_43',
        'user_score_cluster_44',
        'user_score_cluster_45',
        'user_score_cluster_46',
        'user_score_cluster_47',
        'user_score_cluster_48',
        'user_score_cluster_49',
        'user_score_general',
        'total_score_user_question_by_clusters',
        'total_score_user_question_by_clusters_relative']
    return L

main()
