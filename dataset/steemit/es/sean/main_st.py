import argparse
import networkx as nx
import ucb
import timeit
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from article_classifier import shan
from keras.preprocessing.sequence import pad_sequences


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Social Exploration Attention Network.")

    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='validation split. Default is 0.1.')

    parser.add_argument('--walk-length', type=int, default=3,
                        help='Length of walk per source. Default is 3.')

    parser.add_argument('--num-walks', type=int, default=3,
                        help='Number of walks per source. Default is 3.')

    parser.add_argument('--alpha', type=float, default=1,
                        help='Regularization hyperparameter. Default is 1.')

    parser.add_argument(
        '--output-dir',
        nargs='?',
        default='output/',
        help='Output directory.')

    parser.add_argument(
        '--embedding-matrix',
        nargs='?',
        default='embedding_matrix.npy',
        help='Pre-trained embedding matrix.')

    parser.add_argument(
        '--processed-feed',
        nargs='?',
        default='processed_feed/',
        help='Load the preprocessed feed.')

    parser.add_argument('--max-user', type=int, default=1396,
                        help='Max number of users for steemit.')

    parser.add_argument(
        '--week-count',
        type=int,
        default=18,
        help='week count for steemit. Default is 18.')

    parser.add_argument('--max_num_sent', type=int, default=30,
                        help='Max number of sentences in feed. Default is 30.')

    parser.add_argument('--max_len_sent', type=int, default=100,
                        help='Max length for a sentence. Default is 100.')

    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=300,
        help='Number of word embedding dimensions. Default is 300.')

    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of walks per source. Default is 64.')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout for model. Default is 0.2.')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Epochs for training. Default is 3.')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='Minibatch for training. Default is 128.')
    print("type args", vars(parser.parse_args()))
    return parser.parse_args()


def metric_evaluate(
        y_true,
        y_pred,
        results):
    Y_pred = (y_pred > 0.3).astype(np.int32)
    y_f1 = f1_score(y_true, Y_pred)
    y_precision = precision_score(y_true, Y_pred)
    y_recall = recall_score(y_true, Y_pred)
    y_acc = accuracy_score(y_true, Y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    y_auc = auc(fpr, tpr)
    print("Y_acc:", y_acc)
    print("Y_auc:", y_auc)
    print("Y_f1:", y_f1)
    print("Y_precision:", y_precision)
    print("Y_recall:", y_recall)
    results["acc"].append(y_acc)
    results["auc"].append(y_auc)
    results["f1"].append(y_f1)
    results["precision"].append(y_precision)
    results["recall"].append(y_recall)
    return results


def result_printing(results):
    acc_list = results["acc"]
    auc_list = results["auc"]
    f1_list = results["f1"]
    precision_list = results["precision"]
    recall_list = results["recall"]

    avgs = {"acc": 0, "auc": 0, "f1": 0, "precision": 0, "recall": 0}
    for i in range(len(acc_list)):
        print(
            i + 3,
            acc_list[i],
            auc_list[i],
            f1_list[i],
            precision_list[i],
            recall_list[i])
    avgs["acc"] = np.array(acc_list).mean(axis=0)
    avgs["auc"] = np.array(auc_list).mean(axis=0)
    avgs["f1"] = np.array(f1_list).mean(axis=0)
    avgs["precision"] = np.array(precision_list).mean(axis=0)
    avgs["recall"] = np.array(recall_list).mean(axis=0)
    print("average acc:", avgs["acc"])
    print("average auc:", avgs["auc"])
    print("average f1:", avgs["f1"])
    print("average precision:", avgs["precision"])
    print("average recall:", avgs["recall"])
    return avgs


def load_daily_data(sample_dir, week_id):
    out_file_user = sample_dir + 'week_user_' + str(week_id) + '.npy'
    out_file_article = sample_dir + 'week_article_' + str(week_id) + '.npy'
    out_file_label = sample_dir + 'week_label_' + str(week_id) + '.npy'
    out_file_creator = sample_dir + 'week_creator_' + str(week_id) + '.npy'
    users = np.load(out_file_user, allow_pickle=True)
    articles = np.load(out_file_article, allow_pickle=True)
    labels = np.load(out_file_label, allow_pickle=True)
    creators = np.load(out_file_creator, allow_pickle=True)
    return users, articles, labels, creators


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    edgelist = '../karate.edgelist'

    G = nx.read_edgelist(
        edgelist,
        nodetype=int,
        create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G


def build_social_sample(walks, users, articles, labels):
    social_samples = []
    for i in range(len(labels)):
        user_id = users[i][0]
        if user_id in walks:
            for j in range(len(walks[user_id])):
                social_samples.append([
                    walks[user_id][j],
                    articles[i],
                    labels[i],
                    i])
        else:
            social_samples.append([
                [user_id],
                articles[i],
                labels[i],
                i])
    d_vectorized_sample = np.array(social_samples)
    d_user = pad_sequences(d_vectorized_sample[:, 0], maxlen=args.walk_length+1, padding='post')
    d_article = np.rollaxis(np.dstack(list(d_vectorized_sample[:, 1])), -1)
    d_label = pad_sequences(d_vectorized_sample[:, 2], maxlen=1)
    d_id = d_vectorized_sample[:, 3]
    return [d_user, d_article, d_label, d_id]


def personalized_evaluate(
    f1_dict,
    users,
    truth,
    prediction,
):
    prediction = (prediction > 0.5).astype(np.int32)
    truth_dict = {}
    predict_dict = {}
    for i in range(len(users)):
        user = users[i][0]
        if user not in truth_dict:
            truth_dict[user] = []
            predict_dict[user] = []
        truth_dict[user].append(truth[i][0])
        predict_dict[user].append(prediction[i][0])

    for i in range(len(users)):
        user = users[i][0]
        f1_dict[user].append(
            f1_score(
                np.array(
                    truth_dict[user]), np.array(
                    predict_dict[user])))
    return f1_dict


def random_walks(x):
    """
    # fix the bug: loss node 60, 481, 930 in social network
    nodes = list(x.nodes())
    walks = {}
    for node in nodes:
        walks[node] = []
    """
    nodes = [i for i in range(1, args.max_user+1)]
    walks = {}
    for node in nodes:
        walks[node] = []
    for walk_iter in range(args.num_walks):
        for node in nodes:
            walk = list(np.random.randint(1, args.max_user, size=args.walk_length+1))
            walks[node].append(walk)
    return walks


def sean_training(embedding_matrix):

    def compute_average(ids, y):
        pred = []
        votes = {}
        for i in range(len(ids)):
            j = ids[i]
            if j not in votes:
                votes[j] = []
            votes[j].append(y[i])
        for i in range(len(votes)):
            key = list(votes.keys())[i]
            pred.append([np.mean(votes[key])])
        return np.array(pred)

    # initial the walks for every user
    nx_G = read_graph()
    walks = random_walks(nx_G)
    print("initialing ucb...")
    ucb_G = ucb.Graph(nx_G=nx_G, alpha=args.alpha, init_walks=walks)
    init_num_user_explored, init_acc_history, init_walks = ucb_G.init_all()

    vocab_size = embedding_matrix.shape[0]
    classifier = shan(vocab_size=vocab_size,
                      max_num_sent=args.max_num_sent,
                      max_len_sent=args.max_len_sent,
                      embedding_dim=args.embedding_dim,
                      max_user=args.max_user,
                      hidden_size=args.hidden_size,
                      walk_length=args.walk_length+1,
                      dropout=args.dropout,
                      embedding_matrix=embedding_matrix)
    start_time = timeit.default_timer()
    classifier.build_model()
    stop_time = timeit.default_timer()
    print("compiling time:", stop_time - start_time)
    print("Training SEAN...")
    model = classifier.model

    val_results = {
        "acc": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": []}
    test_results = {
        "acc": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": []}
    prediction = {}
    running_time = {}
    creator_hit = {}
    count = 0
    for week in range(args.week_count - 1):
        print()
        print("training week:", week+1, "testing week", week+2)
        # refresh all dictionaries every 30 weeks
        if week % 30 == 0:
            ucb_G.num_user_explored = init_num_user_explored
            ucb_G.acc_history = init_acc_history
            ucb_G.walks = init_walks

        # training-validation data
        train_key = week + 1
        train_val_users, train_val_articles, train_val_labels, train_val_creators = load_daily_data(
            args.processed_feed, train_key)

        indices = np.arange(train_val_users.shape[0])
        np.random.shuffle(indices)
        train_val_users = train_val_users[indices]
        train_val_articles = train_val_articles[indices]
        train_val_labels = train_val_labels[indices]
        num_validation_samples = int(
            args.validation_split *
            train_val_users.shape[0])

        train_users = train_val_users[:-num_validation_samples]
        train_articles = train_val_articles[:-num_validation_samples]
        train_labels = train_val_labels[:-num_validation_samples]
        val_users = train_val_users[-num_validation_samples:]
        val_articles = train_val_articles[-num_validation_samples:]
        val_labels = train_val_labels[-num_validation_samples:]
        print("train count:", len(train_labels))
        if len(train_labels) == 0:
            continue
        print("validation count:", len(val_labels))

        # training
        print("training...")
        if train_key > 1:
            ucb_G.simulate_nodes = list(set([user[0] for user in train_users]))
            # explore friends for training users
            ucb_G.simulate_walks(
                num_walks=args.num_walks,
                walk_length=args.walk_length+1)
            # update the dictionary num_user_explored
            ucb_G.update_freq(ucb_G.walks)

        train_data = build_social_sample(
            ucb_G.walks, train_users, train_articles, train_labels)
        train_social_inputs = train_data[:2]
        train_outputs = train_data[-2]

        start_time = timeit.default_timer()
        model.fit(train_social_inputs,
                  train_outputs,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  shuffle=True,
                  verbose=1)
        stop_time = timeit.default_timer()
        running_time[train_key] = stop_time - start_time
        print("running time:", running_time[train_key])

        # testing
        print("testing...")
        test_key = week + 2
        test_users, test_articles, test_labels, test_creators = load_daily_data(
            args.processed_feed, test_key)
        test_data = build_social_sample(
            ucb_G.walks, test_users, test_articles, test_labels)
        test_social_inputs = test_data[:2]
        y_test = model.predict(
            test_social_inputs,
            batch_size=args.batch_size,
            verbose=1)
        y_test = compute_average(
            test_data[-1], y_test)
        test_results = metric_evaluate(test_labels, y_test, test_results)
        # update dict accuracy_history after testing
        ucb_G.acc_history = personalized_evaluate(
            ucb_G.acc_history, test_users, test_labels, y_test)
        prediction[test_key] = y_test.tolist()
        creator_hit[test_key] = test_creators.tolist()
        print("positive:", list(test_labels).count(1), "negative:", list(test_labels).count(0),
              "selected ratio:", float(list(test_labels).count(1))/float(len(test_labels)))
        count += len(list(test_labels))
    print("validation results:")
    avg_val = result_printing(val_results)
    print("test results:")
    avg_test = result_printing(test_results)

    logs = {
        "param": vars(args),
        "val": val_results,
        "test": test_results,
        "avg_val": avg_val,
        "avg_test": avg_test,
        "running_time": running_time,
        "prediction": prediction,
        "creator_hit": creator_hit}
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = args.output_dir + timestr + '.json'
    with open(output_file_name, "w") as write_file:
        json.dump(logs, write_file)
    print("count:", count)


if __name__ == "__main__":
    args = parse_args()
    f_feed_dir = args.processed_feed
    f_emb_matrix = open(args.embedding_matrix, 'rb')
    embed_matrix = np.load(f_emb_matrix)

    print("SEAN running...")
    sean_training(embed_matrix)
