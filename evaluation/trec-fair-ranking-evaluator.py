# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
    Evaluation script for the TREC Fair Ranking 2019 track.
"""

from collections import defaultdict
from statistics import mean
import os
import json
import csv
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np



######## INPUT HANDLING ########


class FairRankingTask(object):

    def __init__(self, sequence_file, groundtruth_file, group_annotations_file):

        self.sequence = self.load_sequence_data(sequence_file)
        self.groundtruth = self.load_groundtruth_data(groundtruth_file)
        self.document_to_groups = self.load_document_to_groups(group_annotations_file)

        self.all_groups = set([g for groups in self.document_to_groups.values()
            for g in groups])


    def load_sequence_data(self, input_file):
        """
            Returns: a list of triples:
                [(seq_id, q_num, q_id)]
                1. sequence number
                2. query number within the sequence
                3. query id to match with the groundtruth data
        """

        sequence_data = defaultdict(list)
        with open(input_file, 'r') as f_in:
            for row in csv.reader(f_in, delimiter=','):
                seq_id, q_num = row[0].split('.')
                # seq_id, q_num, q_id
                sequence_data[int(seq_id)].append((int(q_num), int(row[1])))
        for seq_id in sequence_data:
            sequence_data[seq_id].sort()
        return sequence_data


    def load_document_to_groups(self, input_file):
        document_to_groups = {}
        with open(input_file, 'r') as f_in:
            for row in csv.reader(f_in, delimiter=','):
                document_to_groups[row[0]] = row[1:]

        return document_to_groups


    def load_groundtruth_data(self, input_file):
        """
            Returns: a dictionary mapping a q_id to
                a dictionary mapping a doc_id to the relevance value (0/1)
                groundtruth_data[q_id][doc_id] = 0 or 1
        """

        groundtruth_data = defaultdict(dict)
        with open(input_file, 'r') as f_in:
            for line in f_in:
                query_data = json.loads(line)
                for rel_data in query_data['documents']:
                    groundtruth_data[query_data['qid']][rel_data['doc_id']]\
                        = rel_data['relevance']
        return groundtruth_data


    @classmethod
    def stopping_probability(cls, relevance):
        """
            From participant instructions: p(s|d)=f(r_d)
                "this is a monotonic transform of that relevance into a probability of beingsatisfied"
            For this eval, we multiply relevance by a constant.
            For future evals, we'll consider makig the stopping probabilities query dependent.
        """
        return 0.5 * relevance


    def groups_exposure(self, seq_id, submission, gamma):
        #   The sums are aggregated differently than in the partcicipant instructions
        #   to avoid iterating multiple times over a single ranking.

        """
            Returns a dictionary mapping a group id to its normalized total esposure
            in the given submission over a sequence of rankings
        """

        g_exps = dict([(g, 0.) for g in self.all_groups])

        for q_num, q_id in self.sequence[seq_id]:
            stopped_until_now = 1.
            for i, doc_id in enumerate(submission.rankings[seq_id][q_num]):
                # for author in self.document_to_authors[doc_id]:
                if doc_id in self.document_to_groups:
                    # logging.warning("Doc %s not not mapped to a group" % doc_id)
                    for g in self.document_to_groups[doc_id]:
                        g_exps[g] += (gamma ** i) * stopped_until_now #* \
                            #FairRankingTask.stopping_probability(self.groundtruth[q_id][doc_id])
                stopped_until_now *= (1 -
                    FairRankingTask.stopping_probability(self.groundtruth[q_id][doc_id]))

        total = sum(g_exps.values())
        return dict([(g, exp / total) for g, exp in g_exps.items()])


    def groups_relevance(self, seq_id, submission):
        #   The sums are aggregated differently than in the partcicipant instructions
        #   to avoid iterating multiple times over a single ranking.

        """
            Returns a dictionary mapping a group id to its normalized total relevance
            in the given submission over a sequence of rankings
        """

        g_rels = dict([(g, 0.) for g in self.all_groups])

        # Important to iterate over our original sequence
        # to avoid evaluating over manipulated sequences
        for q_num, q_id in self.sequence[seq_id]:
            for doc_id in submission.rankings[seq_id][q_num]:
                if doc_id in self.document_to_groups:
                    for g in self.document_to_groups[doc_id]:
                        g_rels[g] += FairRankingTask.stopping_probability(self.groundtruth[q_id][doc_id])

        total = sum(g_rels.values())
        return dict([(g, rel / total) for g, rel in g_rels.items()])



class FairRankingSubmission(object):

    def __init__(self, run_file):
        self.rankings = self.load_submission(run_file)


    def load_submission(self, input_file):
        """
            Returns: a dictionary mapping a q_id to
                a dictionary mapping a doc_id to the relevance value (0/1)
                groundtruth_data[q_id][doc_id] = 0 or 1
        """

        submission_data = defaultdict(dict)
        with open(input_file, 'r') as f_in:
            for line in f_in:
                ranking_data = json.loads(line)
                #note: this strips the original query ID, as it will be read from the groundtruth data
                # to avoid manipulations
                seq_id, q_num = ranking_data['q_num'].split('.')
                submission_data[int(seq_id)][int(q_num)] = ranking_data['ranking']
        return submission_data



######## UTILITY METRICS ########


def expected_utility(ranking, groundtruth, gamma):

    """
        Assumes ranking is a participant-sorted list of documents

    """

    u = 0.
    stopped_until_now = 1.
    for i, doc_id in enumerate(ranking):
        stop_p = FairRankingTask.stopping_probability(groundtruth[doc_id])
        u += (gamma ** i) * stopped_until_now * stop_p
        stopped_until_now *= (1 - stop_p)
    return u


def avg_expected_utility(seq_id, task, submission, gamma):

    """
        Assumes sequence is sorted by sequence number,
        then by the query number within the sequence.
    """

    return sum([expected_utility(
        submission.rankings[seq_id][q_num], task.groundtruth[q_id], gamma)
        for q_num, q_id in task.sequence[seq_id]]) / len(task.sequence[seq_id])


######## FAIRNESS METRICS ########

def l2_loss(seq_id, task, submission, gamma):
    groups_exposure = task.groups_exposure(seq_id, submission, gamma)
    groups_relevance = task.groups_relevance(seq_id, submission)
    return math.sqrt(sum([(groups_exposure[g] - groups_relevance[g])**2
        for g in task.all_groups]))


if __name__ == '__main__':

    """
         python trec-fair-ranking-evaluator.py  \
            --groundtruth_file TREC-Competition-eval-sample-with-rel.json  \
            --query_sequence_file TREC-Competition-eval-seq-5-25000.csv \
            --group_annotations_file group_annotations/article-IMFLevel.csv \
            --group_definition IMFLevel


    """

    parser = argparse.ArgumentParser(description='Evaluate a TREC Fair Ranking submission.')
    parser.add_argument('--groundtruth_file', help='fair ranking ground truth file')
    parser.add_argument('--query_sequence_file', help='fair ranking query sequences file')
    parser.add_argument('--group_annotations_file', help='document group annotations file')
    parser.add_argument('--group_definition', help='keyword defininf group definitions')
    args = parser.parse_args()


    task = FairRankingTask(args.query_sequence_file, args.groundtruth_file, args.group_annotations_file)


    run_files_prefix = 'fairRuns/'
    run_files = [
        # List of files corresponding to the rankings to evaluate
    ]
    run_files_suffix = '.json'

    performance_all_utility = defaultdict(list)
    performance_team_utility = defaultdict(dict)
    performance_all_fairness = defaultdict(list)
    performance_team_fairness = defaultdict(dict)

    for run in run_files:
        submission = FairRankingSubmission(run_files_prefix + run + run_files_suffix)

        for seq_id in task.sequence:
            fairness = l2_loss(seq_id, task, submission, gamma=0.9)
            utility = avg_expected_utility(seq_id, task, submission, gamma=0.9)

            performance_team_fairness[run][seq_id] = fairness
            performance_team_utility[run][seq_id] = utility
            performance_all_fairness[seq_id].append(fairness)
            performance_all_utility[seq_id].append(utility)

    for run in run_files:
        if not os.path.exists('eval_results/%s' % args.group_definition):
            os.makedirs('eval_results/%s' % args.group_definition)
        with open('eval_results/%s/%s' % (args.group_definition, run), 'w') as f_out:
            f_out.write('grouping\trun\tutil-mean\tutil-std\tunfairness-mean\tunfairness-std\n')
            f_out.write('%s\t%s\t%f\t%f\t%f\t%f\n' % (
                args.group_definition,
                run,
                np.mean([performance_team_utility[run][seq_id] for seq_id in sorted(task.sequence)]), # run's utility (mean)
                np.std([performance_team_utility[run][seq_id] for seq_id in sorted(task.sequence)]), # run's utility (std)
                np.mean([performance_team_fairness[run][seq_id] for seq_id in sorted(task.sequence)]), # run's unfairness (mean)
                np.std([performance_team_fairness[run][seq_id] for seq_id in sorted(task.sequence)]) # run's unfairness (std)
                ))


    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(run_files))]
    for i, run in enumerate(run_files):
        plt.scatter(x=[mean(performance_team_fairness[run].values())],
            y=[mean(performance_team_utility[run].values())], c=[colors[i]],
            label=run, s=30)

    plt.legend(loc='lower right', fontsize='xx-small')
    plt.xlabel("Unfairness: L2")
    plt.ylabel("Utility: expected utility")

    # fig = plt.figure()
    if not os.path.exists('plots/%s' % args.group_definition):
        os.makedirs('plots/%s' % args.group_definition)
    plt.savefig('plots/%s/performance-all.pdf' % args.group_definition)
    # plt.show()
