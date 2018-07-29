from keras.layers import Input, Dense, Lambda, Activation
from keras.models import Sequential
from sklearn.preprocessing import normalize
import pandas as pd

PROJECT_DIR = '/Users/Gal/Documents/Repositories/Workshop-in-Data-Science/'

basic_features = ['Score_ans', 'Score_qus']
nlp_features = ['topic_' + str(i) + "_qus" for i in range(10)] + ['topic_' + str(i) + "_ans" for i in range(10)]
time_features = ['diff_percentile_bucket', 'hirerchy']
tags_features = ["cluster_" + str(i) for i in range(50)] + ["user_score_cluster_" + str(i) for i in range(50)]\
                + ["user_score_general","total_score_user_question_by_clusters", "total_score_user_question_by_clusters_relative"]
users_features = ['neg_answers_user', 'neg_questions_user', 'number_of_answers_user', 'out_degree_user', 'question_total_score_user',
                 'answer_total_score_user', 'closeness_centrality_user', 'in_degree_user']

all_features = basic_features + users_features + time_features  + nlp_features # + tags_features


def main():
    target = 'IsAcceptedAnswer'

    train = pd.read_csv(PROJECT_DIR + "\\data\\clean_data\\train_with_tag_clusters_and_user_scores.csv")
    test = pd.read_csv(PROJECT_DIR + "\\data\\clean_data\\test_with_tag_clusters_and_user_scores.csv")
    
    feature_list = all_features    
    num_features = len(feature_list)

    batch_size = 10
    hidden_dim1 = 50
    hidden_dim2 = 50
    epochs = 20
    print("batch_size=", batch_size, "\nepochs=", epochs, "\nhidden_dims=", [hidden_dim1, hidden_dim2])
    # create model:
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

    model.fit(train[feature_list], train[target].astype(int), epochs=epochs, batch_size=batch_size, verbose=0)

    loss_and_metrics = model.evaluate(test[feature_list], test[target].astype(int), batch_size=batch_size)
    print("\n---------------\nloss metrices:")
    print(loss_and_metrics)

    scores = model.evaluate(test[feature_list], test[target])
    print("\n---------------\nscores:")
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__ == "__main__":
    main()
