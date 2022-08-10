import torch
import numpy as np
from collections import defaultdict
from model import PathCon
from utils import sparse_to_tuple
import pandas as pd

args = None


def train(model_args, data):
    global args, model, sess
    args = model_args

    # extract data
    triplets, paths, n_relations, n_attributes, neighbor_params, path_params = data

    train_triplets, valid_triplets, test_triplets = triplets
    train_edges = torch.LongTensor(np.array(range(len(train_triplets)), np.int32))
    train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32))
    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))

    train_paths, valid_paths, test_paths = paths

    train_labels = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    # define the model
    model = PathCon(args, n_relations, n_attributes, neighbor_params, path_params)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    print('start training ...')

    epoch_result_list = []
    train_result_list = []
    valid_result_list = []
    test_result_list = []
    mrr_result_list = []
    mr_result_list = []
    h1_result_list = []
    h3_result_list = []
    h5_result_list = []
    for step in range(args.epoch):
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        if args.use_context:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]
        if args.use_path:
            train_paths = train_paths[index]  # path tensor
        train_labels = train_labels[index]  # relation tensor

        # training
        s = 0
        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_paths, train_labels, s, s + args.batch_size))
            # print(s, " to ", s + args.batch_size, "  loss:", loss)
            s += args.batch_size

        # evaluation
        print('epoch %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_paths, train_labels)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_paths, valid_labels)
        test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

        # show evaluation result for current epoch
        current_res = 'acc: %.4f' % test_acc
        print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5)
        print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5))
        print()
        epoch_result_list.append(step)
        train_result_list.append(train_acc)
        valid_result_list.append(valid_acc)
        test_result_list.append(test_acc)
        mrr_result_list.append(mrr)
        mr_result_list.append(mr)
        h1_result_list.append(hit1)
        h3_result_list.append(hit3)
        h5_result_list.append(hit5)

        # update final results according to validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    # show final evaluation result
    print('final results\n%s' % final_res)

    print('writing result to csv...')
    dataframe = pd.DataFrame({'step': epoch_result_list,
                              'train_acc': train_result_list,
                              'valid_acc': valid_result_list,
                              'test_acc': test_result_list,
                              'mrr': mrr_result_list,
                              'mr': mr_result_list,
                              'hit1': h1_result_list,
                              'hit3': h3_result_list,
                              'hit5': h5_result_list
                              })

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("../src/result/new_model_result.csv", index=False, sep=',')



def get_feed_dict(entity_pairs, train_edges, paths, labels, start, end):
    feed_dict = {}

    if args.use_context:
        feed_dict["entity_pairs"] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict["train_edges"] = train_edges[start:end]
        else:
            # for evaluation no edges should be masked out
            feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                        else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    if args.use_path:
        if args.path_type == 'embedding':
            indices, values, shape = sparse_to_tuple(paths[start:end])
            indices = torch.LongTensor(indices).cuda() if args.cuda else torch.LongTensor(indices)
            values = torch.Tensor(values).cuda() if args.cuda else torch.Tensor(values)
            feed_dict["path_features"] = torch.sparse.FloatTensor(indices.t(), values, torch.Size(shape)).to_dense()
        elif args.path_type == 'rnn':
            feed_dict["path_ids"] = torch.LongTensor(paths[start:end]).cuda() if args.cuda \
                    else torch.LongTensor(paths[start:end])

    feed_dict["labels"] = labels[start:end]

    return feed_dict


def evaluate(entity_pairs, paths, labels):
    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.test_step(model, get_feed_dict(
            entity_pairs, None, paths, labels, s, s + args.batch_size))
        acc_list.extend(acc)
        scores_list.extend(scores)
        s += args.batch_size

    return float(np.mean(acc_list)), np.array(scores_list)


def calculate_ranking_metrics(triplets, scores, true_relations):
    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations = np.array(triplets)[0:scores.shape[0], 2]
    relation_expended = np.expand_dims(relations, 1)
    sorted_indices -= relation_expended
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5
