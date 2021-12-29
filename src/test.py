# Testing sctipt 
# Originally Forked from https://github.com/ExplorerFreda/VGNSL?files=1
# Modified by Noriyuki Kojima (nk654@cornell.edu)

# Usage Example
#  CUDA_VISIBLE_DEVICES=0 python test.py --candidate path_to_checkpoint --record_trees --splits test

import argparse
import os
import numpy as np
import pickle
import operator

from evaluation import test_trees
from vocab import Vocabulary

from nltk.tree import Tree


# This should keeps the order of low-order span to higher-order spans
def extract_spans(tree):
    answer = list()
    stack = list()
    items = tree.split()
    curr_index, error_ct = 0, 0
    for item in items:
        try:
            if item == ')':
                pos = -1
                right_margin = stack[pos][1]
                left_margin = None
                while stack[pos] != '(':
                    left_margin = stack[pos][0]
                    pos -= 1
                assert left_margin is not None
                assert right_margin is not None
                stack = stack[:pos] + [(left_margin, right_margin)]
                answer.append((left_margin, right_margin))
            elif item == '(':
                stack.append(item)
            else:
                stack.append((curr_index, curr_index))
                curr_index += 1
        except:
            error_ct += 1
            #print("ERROR {}: something went wrong during the evaluation.".format(error_ct))
            return answer
    return answer


def extract_spans_tokens(tree, concat=False):
    answer = list()
    stack = list()
    items = tree.split()
    tokens = tree.replace("(", "").replace(")", "").split()
    curr_index = 0
    for item in items:
        if item == ')':
            pos = -1
            right_margin = stack[pos][1]
            left_margin = None
            while stack[pos] != '(':
                left_margin = stack[pos][0]
                pos -= 1
            assert left_margin is not None
            assert right_margin is not None
            stack = stack[:pos] + [(left_margin, right_margin)]
            if concat == True:
                span = " ".join(tokens[left_margin:right_margin+1])
                answer.append(span)
            else:
                answer.append(tokens[left_margin:right_margin+1])
        elif item == '(':
            stack.append(item)
        else:
            stack.append((curr_index, curr_index))
            curr_index += 1
    return answer


def extract_tokens(tree, margins):
    spans = []
    tokens = tree.replace("(", "").replace(")", "").split()
    for m in margins:
        left_margin, right_margin = m
        spans.append(tokens[left_margin:right_margin+1])
    return spans


def extract_statistics(gold_tree_spans, produced_tree_spans, gap_stats=True):
    gold_tree_spans = set(gold_tree_spans)
    produced_tree_spans = set(produced_tree_spans)
    precision_cnt = sum(list(
        map(lambda span: 1.0 if span in gold_tree_spans else 0.0, produced_tree_spans)))
    recall_cnt = sum(list(
        map(lambda span: 1.0 if span in produced_tree_spans else 0.0, gold_tree_spans)))
    precision_denom = len(produced_tree_spans)
    recall_denom = len(gold_tree_spans)
    if gap_stats:
        not_recovered = list(map(
            lambda span: span if span not in produced_tree_spans else None, gold_tree_spans))
        recovered = list(
            map(lambda span: span if span in produced_tree_spans else None, gold_tree_spans))
        return precision_cnt, precision_denom, recall_cnt, recall_denom, not_recovered, recovered
    else:
        return precision_cnt, precision_denom, recall_cnt, recall_denom


def f1_score(produced_trees, orig_gold_trees, splits=None, ctg_eval=False, gap_stats=True, eps=1e-6):
    gold_spans = list(
        map(lambda tree: extract_spans_tokens(tree, True), orig_gold_trees))
    produced_spans = list(
        map(lambda tree: extract_spans_tokens(tree, True), produced_trees))
    gold_trees = list(map(lambda tree: extract_spans(tree), orig_gold_trees))
    produced_trees = list(map(lambda tree: extract_spans(tree), produced_trees))
    assert len(produced_trees) == len(gold_trees)
    precision_cnt, precision_denom, recall_cnt, recall_denom = 0, 0, 0, 0

    # F1-statistics
    for i, item in enumerate(produced_trees):
        if gap_stats:
            pc, pd, rc, rd, _, _ = extract_statistics(
                gold_trees[i], item, gap_stats=gap_stats)

        else:
            pc, pd, rc, rd = extract_statistics(
                gold_trees[i], item, gap_stats=gap_stats)
        precision_cnt += pc
        precision_denom += pd
        recall_cnt += rc
        recall_denom += rd

    # Recall 
    # TODO: change path
    if ctg_eval:
        fail_ct = 0
        in_name = "/home/noriyuki/Desktop/VGNSL/data/mscoco/{}_labels.npy".format(
            splits)
        categs = pickle.load(open(in_name, "rb"))
        np_recall_cnt, vp_recall_cnt, pp_recall_cnt, adjp_recall_cnt = 0, 0, 0, 0
        np_recall_denom, vp_recall_denom, pp_recall_denom, adjp_recall_denom = 0, 0, 0, 0
        all_adj_gold_trees = []
        np_gaps, pp_gaps, vp_gaps, adj_gaps = {}, {}, {}, {}
        np_recs, pp_recs, vp_recs, adj_recs = {}, {}, {}, {}

        for i, item in enumerate(produced_trees):
            np_gold_trees, pp_gold_trees, vp_gold_trees, adj_gold_trees = [], [], [], []
            for j, t in enumerate(gold_spans[i]):
                try:
                    ctg = categs[i][t]
                    if ctg == "NP":  # or ctg == "NX":
                        np_gold_trees.append(gold_trees[i][j])
                    elif ctg == "PP":
                        pp_gold_trees.append(gold_trees[i][j])
                    elif ctg == "VP":
                        vp_gold_trees.append(gold_trees[i][j])
                    elif ctg == "ADJP":
                        adj_gold_trees.append(gold_trees[i][j])
                        all_adj_gold_trees.append(t)
                except:
                    print("Something went wrong.")
                    print(t)
                    fail_ct += 1

            # NP
            if gap_stats:
                pc, pd, rc, rd, gap, rec = extract_statistics(
                    np_gold_trees, item, gap_stats=gap_stats)
                np_gaps[i] = []
                for g in gap:
                    if g is not None:
                        np_gaps[i].append(g)
                np_recs[i] = []
                for r in rec:
                    if r is not None:
                        np_recs[i].append(r)
            else:
                pc, pd, rc, rd = extract_statistics(
                    np_gold_trees, item, gap_stats=gap_stats)
            np_recall_cnt += rc
            np_recall_denom += rd
            # VP
            if gap_stats:
                pc, pd, rc, rd, gap, rec = extract_statistics(
                    vp_gold_trees, item, gap_stats=gap_stats)
                vp_gaps[i] = []
                for g in gap:
                    if g is not None:
                        vp_gaps[i].append(g)
                vp_recs[i] = []
                for r in rec:
                    if r is not None:
                        vp_recs[i].append(r)
            else:
                pc, pd, rc, rd = extract_statistics(
                    vp_gold_trees, item, gap_stats=gap_stats)
            vp_recall_cnt += rc
            vp_recall_denom += rd
            # PP
            if gap_stats:
                pc, pd, rc, rd, gap, rec = extract_statistics(
                    pp_gold_trees, item, gap_stats=gap_stats)
                pp_gaps[i] = []
                for g in gap:
                    if g is not None:
                        pp_gaps[i].append(g)
                pp_recs[i] = []
                for r in rec:
                    if r is not None:
                        pp_recs[i].append(r)
            else:
                pc, pd, rc, rd = extract_statistics(
                    pp_gold_trees, item, gap_stats=gap_stats)
            pp_recall_cnt += rc
            pp_recall_denom += rd
            # ADJP
            if gap_stats:
                pc, pd, rc, rd, gap, rec = extract_statistics(
                    adj_gold_trees, item, gap_stats=gap_stats)
                adj_gaps[i] = []
                for g in gap:
                    if g is not None:
                        adj_gaps[i].append(g)
                adj_recs[i] = []
                for r in rec:
                    if r is not None:
                        adj_recs[i].append(r)
            else:
                pc, pd, rc, rd = extract_statistics(
                    adj_gold_trees, item, gap_stats=gap_stats)
            adjp_recall_cnt += rc
            adjp_recall_denom += rd

        npr = np.round((np_recall_cnt / float(np_recall_denom+eps)) * 100, 3)
        vpr = np.round((vp_recall_cnt / float(vp_recall_denom+eps)) * 100, 3)
        ppr = np.round((pp_recall_cnt / float(pp_recall_denom+eps)) * 100, 3)
        adjpr = np.round(
            (adjp_recall_cnt / float(adjp_recall_denom+eps)) * 100, 3)
        print("Failed samples {}".format(fail_ct))
        np_perc = np.round((np_recall_denom / recall_denom+eps) * 100, 3)
        vp_perc = np.round((vp_recall_denom / recall_denom+eps) * 100, 3)
        pp_perc = np.round((pp_recall_denom / recall_denom+eps) * 100, 3)
        adj_perc = np.round((adjp_recall_denom / recall_denom+eps) * 100, 3)
        print("NP Recall :{} ({} %), VP Recall: {} ({} %), PP Recall: {} ({} %), ADJP Recall: {} ({} %)".format(npr, np_perc, vpr, vp_perc,
                                                                                                                ppr, pp_perc, adjpr, adj_perc))
    precision = float(precision_cnt) / (precision_denom+eps) * 100.0
    recall = float(recall_cnt) / (recall_denom+eps) * 100.0
    f1 = 2 * precision * recall / (precision + recall + eps)

    if gap_stats:
        return f1, precision, recall, (np_gaps, pp_gaps, vp_gaps, adj_gaps), (np_recs, pp_recs, vp_recs, adj_recs)
    else:
        return f1, precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=str,
                        required=True, help='model path to evaluate')
    parser.add_argument('--splits', type=str, default='test', help='val | test')
    parser.add_argument('--record_trees', action='store_true', help='record predict trees to a text file')
    parser.add_argument('--ctg_eval', action='store_true', help='calculate category recall')
    parser.add_argument('--self_f1', action='store_true', help='self-F1 checkpoint selection')
    parser.add_argument('--gtfile', type=str, default="", help='path to predicition result for self-F1 checkpoint selection')

    args = parser.parse_args()

    if args.self_f1:
        # self-F1 evalutation for checkpoint selection 
        trees, ground_truth, ordered_spans = test_trees(
            args.candidate, args.splits, args.gtfile, return_indicies=True)
    else:
        # evaluation
        trees, ground_truth, ordered_spans = test_trees(
            args.candidate, args.splits, return_indicies=True)

    if args.record_trees:
        from datetime import date
        today = date.today()
        folder_name = "./outputs/trees/" + str(today)
        os.makedirs(folder_name, exist_ok=True)
        file_name = "-".join(args.candidate.split("/")
                             [-2:]).replace("/", "-") + ".txt"
        outpath = "{}/{}".format(folder_name, file_name)
        outfile = open(outpath, "w")
        for tree in trees:
            outfile.write(tree)
            outfile.write("\n")
        outfile.close()

    f1, _, _ = f1_score(trees, ground_truth, args.splits,
                        ctg_eval=False, gap_stats=False)
    print('Model:', args.candidate)
    print('F1 score:', f1)