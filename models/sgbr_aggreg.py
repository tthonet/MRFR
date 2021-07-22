import numpy as np
import json
import csv
import itertools
import argparse
import os
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gamma", type=float, help="Continuation probability", default=0.9)
parser.add_argument("-l", "--lambd", type=float, help="Weight of fairness wrt utility", default=1.0)
parser.add_argument("-t", "--beta", type=float, help="Weight of deserved exposure wrt rel in init sorting", default=1.0)
parser.add_argument("-e", "--eps", type=float, help="Probability of satisfaction given a relevant document", default=0.5)
parser.add_argument("-b", "--bin_size", type=int, help="Size of the bins of documents", default=3)
parser.add_argument("-n", "--max_n_bins", type=int, help="Maximum number of bins", default=1)
parser.add_argument("-f", "--grouping_file", type=str, help="Grouping files for fairness definition, space-separated")
args = parser.parse_args()

EPS = args.eps # Probability for the user to be satisfied given that the document is relevant
GAMMA = args.gamma # Continuation probability
LAMBDA = args.lambd # Tradeoff hyperparameter balancing utility and fairness
BETA = args.beta # Tradeoff hyperparameter balancing relevance and deserved exposure in the initial sorting of docs
BIN_SIZE = args.bin_size # Size of the bins of documents to split the rankings
MAX_N_BINS = args.max_n_bins # Maximum number of bins to permute over in the rankings (remaining bins are kept fixed)
NOISY_REL = args.noisy_rel # Indicates whether noisy version of relevance score is used

def exposure(ranking, n_groups, doc_groups, rel):
    exps = [0.0] * n_groups
    prod = 1
    for i in range(len(ranking)):
        doc = ranking[i]
        if len(doc_groups[doc]) > 0:
            for group in doc_groups[doc]:
                exps[group] += (GAMMA ** i) * prod
        prod *= 1 - probability(doc, rel)
    return exps

def relevance(ranking, n_groups, doc_groups, rel):
    rels = [0.0] * n_groups
    for i in range(len(ranking)):
        doc = ranking[i]
        if len(doc_groups[doc]) > 0:
            for group in doc_groups[doc]:
                rels[group] += probability(doc, rel)
    return rels

def utility(ranking, rel):
    sum = 0
    prod = 1
    for i in range(len(ranking)):
        doc = ranking[i]
        sum += (GAMMA ** i) * prod * probability(doc, rel)
        prod *= 1 - probability(doc, rel)
    return sum

def probability(doc, rel):
    return EPS * rel[doc]

# Load the queries and documents
dir_path = "../data/"
query_file_path = dir_path + "fair-TREC-evaluation-sample.json"
query_file = open(query_file_path, "r", encoding="utf8")
queries = [json.loads(line) for line in query_file.readlines()]
queries = {query['qid']: query for query in queries} # The data of unique queries
sequence_file_path = dir_path + "fair-TREC-evaluation-sequences.csv" #"fair-TREC-evaluation-sequences-test.csv"
sequence_file = open(sequence_file_path, "r", encoding="utf8")
sequences = list(csv.reader(sequence_file, delimiter=',')) # The sequences of searches (w/ potentially repeated queries)

# Load the group partition according to which fairness is defined
grouping_file_names = args.grouping_file.split(" ")
partition_file_paths = [dir_path + "groupings-target/" + grouping_file_name for grouping_file_name in grouping_file_names]
partition_files = [open(partition_file_path, "r", encoding="utf8") for partition_file_path in partition_file_paths]
global_doc_groups = {}
for i, partition_file in enumerate(partition_files):
    global_doc_groups[i] = {}
    for row in csv.reader(partition_file, delimiter=','):
        global_doc_groups[i][row[0]] = row[1:]
    partition_file.close()
n_partitions = len(global_doc_groups)

# Find a (potentially sub-)optimal ranking for every query
exposure_history = {} # Exposure history per group for each query, used for amortization in repeated queries
relevance_history = {} # Relevance history per group for each query, used for amortization in repeated queries
utility_history = {} # Utility history for each query, used for amortization in repeated queries
discrepancy_history = {} # Discrepancy history for each query
sequence_query_counts = {} # Number of time per sequence each query occurs in different searches
sequence_query_eval_scores = {} # Evaluation scores for each unique query, with amortization for repeated queries
search_rankings = [] # The rankings obtained for each search
init_time = time()
for search in sequences:
    query_id = int(search[1])
    query = queries[query_id]
    [sequence_id, search_id] = search[0].split(".")
    if int(search_id) % 1000 == 0:
        print("Processing sequence %s -- search %s -- time %.3fs" % (sequence_id, search_id, time() - init_time),
              flush=True)

    # Fetch relevant information in the query
    query_docs = query['documents'] # Documents associated with the query
    doc_groups = [[] for _ in range(n_partitions)]
    query_group_dict = [{} for _ in range(n_partitions)] # Mapping between global group ids and query group ids
    rel_probas = []
    filtered_query_docs = []
    n_groups = [0 for _ in range(n_partitions)]
    for (i, query_doc) in enumerate(query_docs):
        query_doc_id = query_doc['doc_id']
        filtered_query_docs.append(query_doc)
        if NOISY_REL:
            rel_probas.append(query_doc['noisy_rel'])  # Use the groundtruth relevance
        else:
            rel_probas.append(query_doc['relevance'])  # Use the groundtruth relevance

        # Group assignments in the document
        for partition in range(n_partitions):
            current_doc_groups = []
            if query_doc_id in global_doc_groups[partition]:
                for group in global_doc_groups[partition][query_doc_id]:
                    if group not in query_group_dict[partition]:
                        query_group_dict[partition][group] = n_groups[partition]
                        n_groups[partition] = n_groups[partition] + 1
                    current_doc_groups.append(query_group_dict[partition][group])
            #doc_groups.append(set(current_doc_groups)) # Repetitions of the same group in doc contribute only once
            doc_groups[partition].append(current_doc_groups) # Repetitions of the same group in doc all contribute
    n_query_docs = len(filtered_query_docs)

    # Find a (potentially sub-)optimal ranking for this query
    ## Fetch the history of exposure and relevance if the query has already been processed
    if sequence_id in sequence_query_counts:
        if query_id in sequence_query_counts[sequence_id]:
            cumul_exposures = [exposure_history[sequence_id][query_id][partition] for partition in range(n_partitions)]
            cumul_relevances = [relevance_history[sequence_id][query_id][partition] for partition in range(n_partitions)]
            cumul_utilities = utility_history[sequence_id][query_id]
            query_count = sequence_query_counts[sequence_id][query_id]
        else: # First time the query is processed
            cumul_exposures = [[0.0] * n_groups[partition] for partition in range(n_partitions)]
            cumul_relevances = [[0.0] * n_groups[partition] for partition in range(n_partitions)]
            cumul_utilities = 0.0
            query_count = 0
    else: # First time the sequence is processed
        exposure_history[sequence_id] = {}
        relevance_history[sequence_id] = {}
        utility_history[sequence_id] = {}
        discrepancy_history[sequence_id] = {}
        sequence_query_counts[sequence_id] = {}
        sequence_query_eval_scores[sequence_id] = {}
        cumul_exposures = [[0.0] * n_groups[partition] for partition in range(n_partitions)]
        cumul_relevances = [[0.0] * n_groups[partition] for partition in range(n_partitions)]
        cumul_utilities = 0.0
        query_count = 0

    ## Sort the documents based on relevance and discrepancy
    ### Sort the SORT_CUTOFF top relevance documents
    rel_sort_scores = np.asarray(rel_probas)
    ### Sort the remaining documents according to deserved exposure in descending order, to identify documents with a
    ### higher potential to reduce the query discrepancy after they are re-ranked
    exposure_norm = [sum(cumul_exposures[partition]) for partition in range(n_partitions)]
    relevance_norm = [sum(cumul_relevances[partition]) for partition in range(n_partitions)]
    if sum(exposure_norm) > 0 and sum(relevance_norm) > 0:
        de_sort_scores = [] # Deserved exposure scores
        for id in range(n_query_docs):
            doc_discrepancy = 0.0
            for partition in range(n_partitions):
                for g in doc_groups[partition][id]:
                    past_exposure = cumul_exposures[partition][g] / exposure_norm[partition]
                    past_relevance = cumul_relevances[partition][g] / relevance_norm[partition]
                    doc_discrepancy += past_exposure - past_relevance # Past over-exposure
            de_sort_scores.append(doc_discrepancy / n_partitions)
    else:
        de_sort_scores = [0.0] * n_query_docs # If no docs are relevant, consider that all docs get deserved exposure
    de_sort_scores = np.asarray(de_sort_scores)
    ### Split the documents into bins
    sort_scores = rel_sort_scores - BETA * de_sort_scores
    sorted_doc_ids = np.argsort(-sort_scores) # Sort by combination of rel and (negative) over-exposure
    doc_bins = [sorted_doc_ids[i:i + BIN_SIZE] for i in range(0, len(sorted_doc_ids), BIN_SIZE)]

    ## Generate the rankings as all the possible combinations of the bins
    bin_permutations = [list(itertools.permutations(doc_bin)) for doc_bin in doc_bins]
    rankings = [[]]
    for i in range(len(bin_permutations)):
        if i < MAX_N_BINS:
            rankings = [ranking + list(bin_permutations[i][j]) for ranking in rankings
                        for j in range(len(bin_permutations[i]))]
        else:
            rankings = [ranking + list(doc_bins[i]) for ranking in rankings]
    n_rankings = len(rankings)

    ## Evaluate every ranking
    optimal_ranking_id = -1
    optimal_ranking_score = -np.inf
    optimal_ranking_group_exposures = []
    optimal_ranking_group_relevances = []
    optimal_ranking_utility = 0.0
    optimal_ranking_discrepancy = 0.0
    for r in range(n_rankings):
        ranking = rankings[r]

        # Compute the ranking's utility
        util = 1.0 / (query_count + 1) * (utility(ranking, rel_probas) + query_count * cumul_utilities)

        # Compute the normalization constant for exposure
        current_exposures = [exposure(ranking, n_groups[partition], doc_groups[partition], rel_probas)
                             for partition in range(n_partitions)]
        group_exposures = [[current_exposures[partition][g] + cumul_exposures[partition][g]
                            for g in range(n_groups[partition])] for partition in range(n_partitions)]
        exposure_norm = [sum(group_exposures[partition]) for partition in range(n_partitions)]

        # Compute the normalization constant for relevance
        current_relevances = [relevance(ranking, n_groups[partition], doc_groups[partition], rel_probas)
                              for partition in range(n_partitions)]
        group_relevances = [[current_relevances[partition][g] + cumul_relevances[partition][g]
                             for g in range(n_groups[partition])] for partition in range(n_partitions)]
        relevance_norm = [sum(group_relevances[partition]) for partition in range(n_partitions)]

        # Compute the discrepancy for the current partition
        discrepancy = 0.0
        if sum(exposure_norm) > 0 and sum(relevance_norm) > 0:
            for partition in range(n_partitions):
                partition_discrepancy = 0.0
                for g in range(n_groups[partition]):
                    amortized_exposure = group_exposures[partition][g] / exposure_norm[partition]
                    amortized_relevance = group_relevances[partition][g] / relevance_norm[partition]
                    partition_discrepancy += np.square(amortized_exposure - amortized_relevance)
                discrepancy += np.sqrt(partition_discrepancy)
            discrepancy = discrepancy / n_partitions

        # Compute the overall score for the current ranking on the current partition
        eval_score = util - LAMBDA * discrepancy

        if eval_score > optimal_ranking_score:
            optimal_ranking_id = r
            optimal_ranking_score = eval_score
            optimal_ranking_group_exposures = group_exposures
            optimal_ranking_group_relevances = group_relevances
            optimal_ranking_utility = util
            optimal_ranking_discrepancy = discrepancy
    sequence_query_eval_scores[sequence_id][query_id] = optimal_ranking_score
    optimal_ranking = [filtered_query_docs[d]['doc_id'] for d in rankings[optimal_ranking_id]]
    search_rankings.append({'q_num': sequence_id + "." + search_id, 'qid': query_id, 'ranking': optimal_ranking})

    # Update history variables
    exposure_history[sequence_id][query_id] = optimal_ranking_group_exposures
    relevance_history[sequence_id][query_id] = optimal_ranking_group_relevances
    utility_history[sequence_id][query_id] = optimal_ranking_utility
    discrepancy_history[sequence_id][query_id] = optimal_ranking_discrepancy
    sequence_query_counts[sequence_id][query_id] = query_count + 1

elapsed_time = time() - init_time
print("Elapsed time (s):", elapsed_time, flush=True)
mean_utility = np.mean([np.mean(list(query_utilities.values()))
                        for query_utilities in utility_history.values()])
mean_discrepancy = np.mean([np.mean(list(query_discrepancies.values()))
                            for query_discrepancies in discrepancy_history.values()])
mean_eval_score = np.mean([np.mean(list(query_eval_scores.values()))
                           for query_eval_scores in sequence_query_eval_scores.values()])

query_file.close()
sequence_file.close()

output_path = dir_path + "results/all_CRP_T_groupings_8/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save the obtained rankings
ranking_file_path = output_path + "evaluation-rankings-aggreg-g" + str(GAMMA) + "-l" + str(LAMBDA) + "-b" + \
                    str(BIN_SIZE) + "-n" + str(MAX_N_BINS) + "-t" + str(BETA) + "-e" + str(EPS) + ".json"

with open(ranking_file_path, "w", encoding="utf8") as ranking_file:
    for search_ranking in search_rankings:
        ranking_file.write(json.dumps(search_ranking) + "\n")

# Save the log
log_file_path = output_path + "evaluation-log-aggreg-g" + str(GAMMA) + "-l" + str(LAMBDA) + "-b" + \
                str(BIN_SIZE) + "-n" + str(MAX_N_BINS) + "-t" + str(BETA) + "-e" + str(EPS) + ".txt"

with open(log_file_path, "w", encoding="utf8") as log_file:
    log_file.write("Mean utility: " + str(mean_utility) + "\n")
    log_file.write("Mean discrepancy: " + str(mean_discrepancy) + "\n")
    log_file.write("Mean eval score: " + str(mean_eval_score) + "\n")
    log_file.write("Elapsed time (s): " + str(elapsed_time) + "\n")
    log_file.write("Sequence-specific utility per query: " + "\n")
    log_file.write(str(utility_history) + "\n")
    log_file.write("Sequence-specific discrepancy per query: " + "\n")
    log_file.write(str(discrepancy_history) + "\n")
    log_file.write("Sequence-specific eval score per query: " + "\n")
    log_file.write(str(sequence_query_eval_scores) + "\n")
