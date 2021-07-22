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
parser.add_argument("-f", "--grouping_file", type=str, help="Grouping file for fairness definition")
args = parser.parse_args()

EPS = args.eps # Probability for the user to be satisfied given that the document is relevant
GAMMA = args.gamma # Continuation probability
LAMBDA = args.lambd # Tradeoff hyperparameter balancing utility and fairness
BETA = args.beta # Tradeoff hyperparameter balancing relevance and deserved exposure in the initial sorting of docs
BIN_SIZE = args.bin_size # Size of the bins of documents to split the rankings
MAX_N_BINS = args.max_n_bins # Maximum number of bins to permute over in the rankings (remaining bins are kept fixed)

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
sequence_file_path = dir_path + "fair-TREC-evaluation-sequences.csv"
sequence_file = open(sequence_file_path, "r", encoding="utf8")
sequences = list(csv.reader(sequence_file, delimiter=',')) # The sequences of searches (w/ potentially repeated queries)

# Load the group partition according to which fairness is defined
partition_file_path = dir_path + "groupings-source/" + args.grouping_file
partition_file = open(partition_file_path, "r", encoding="utf8")
global_doc_groups = {}
for row in csv.reader(partition_file, delimiter=','):
    global_doc_groups[row[0]] = row[1:]
partition_file.close()

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
    doc_groups = []
    query_group_dict = {} # Mapping between global group ids and query group ids
    rel_probas = []
    filtered_query_docs = []
    n_groups = 0
    for (i, query_doc) in enumerate(query_docs):
        query_doc_id = query_doc['doc_id']
        filtered_query_docs.append(query_doc)
        rel_probas.append(query_doc['relevance']) # Use the groundtruth relevance

        # Group assignments in the document
        current_doc_groups = []
        if query_doc_id in global_doc_groups:
            for group in global_doc_groups[query_doc_id]:
                if group not in query_group_dict:
                    query_group_dict[group] = n_groups
                    n_groups = n_groups + 1
                current_doc_groups.append(query_group_dict[group])
        #doc_groups.append(set(current_doc_groups)) # Repetitions of the same group in doc contribute only once
        doc_groups.append(current_doc_groups) # Repetitions of the same group in doc all contribute
    n_query_docs = len(filtered_query_docs)

    # Find a (potentially sub-)optimal ranking for this query
    ## Fetch the history of exposure and relevance if the query has already been processed
    if sequence_id in sequence_query_counts:
        if query_id in sequence_query_counts[sequence_id]:
            cumul_exposures = exposure_history[sequence_id][query_id]
            cumul_relevances = relevance_history[sequence_id][query_id]
            cumul_utilities = utility_history[sequence_id][query_id]
            query_count = sequence_query_counts[sequence_id][query_id]
        else: # First time the query is processed
            cumul_exposures = [0.0] * n_groups
            cumul_relevances = [0.0] * n_groups
            cumul_utilities = 0.0
            query_count = 0
    else: # First time the sequence is processed
        exposure_history[sequence_id] = {}
        relevance_history[sequence_id] = {}
        utility_history[sequence_id] = {}
        discrepancy_history[sequence_id] = {}
        sequence_query_counts[sequence_id] = {}
        sequence_query_eval_scores[sequence_id] = {}
        cumul_exposures = [0.0] * n_groups
        cumul_relevances = [0.0] * n_groups
        cumul_utilities = 0.0
        query_count = 0

    ## Sort the documents based on relevance and discrepancy
    ### Sort the SORT_CUTOFF top relevance documents
    rel_sort_scores = np.asarray(rel_probas)
    ### Sort the remaining documents according to deserved exposure in descending order, to identify documents with a
    ### higher potential to reduce the query discrepancy after they are re-ranked
    exposure_norm = sum(cumul_exposures)
    relevance_norm = sum(cumul_relevances)
    if exposure_norm > 0 and relevance_norm > 0:
        de_sort_scores = [] # Deserved exposure scores
        for id in range(n_query_docs):
            doc_discrepancy = 0.0
            for g in doc_groups[id]:
                past_exposure = cumul_exposures[g] / exposure_norm
                past_relevance = cumul_relevances[g] / relevance_norm
                doc_discrepancy += past_exposure - past_relevance # Past over-exposure
            de_sort_scores.append(doc_discrepancy)
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
        current_exposures = exposure(ranking, n_groups, doc_groups, rel_probas)
        group_exposures = [current_exposures[g] + cumul_exposures[g] for g in range(n_groups)]
        exposure_norm = sum(group_exposures)

        # Compute the normalization constant for relevance
        current_relevances = relevance(ranking, n_groups, doc_groups, rel_probas)
        group_relevances = [current_relevances[g] + cumul_relevances[g] for g in range(n_groups)]
        relevance_norm = sum(group_relevances)

        # Compute the discrepancy for the current partition
        discrepancy = 0.0
        if exposure_norm > 0 and relevance_norm > 0:
            for g in range(n_groups):
                amortized_exposure = group_exposures[g] / exposure_norm
                amortized_relevance = group_relevances[g] / relevance_norm
                discrepancy += np.square(amortized_exposure - amortized_relevance)
            discrepancy = np.sqrt(discrepancy)

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

output_path = dir_path + "results/" + args.grouping_file + "/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save the obtained rankings
ranking_file_path = output_path + "evaluation-rankings-single-g" + str(GAMMA) + "-l" + str(LAMBDA) + "-b" + \
                    str(BIN_SIZE) + "-n" + str(MAX_N_BINS) + "-t" + str(BETA) + "-e" + str(EPS) + ".json"

with open(ranking_file_path, "w", encoding="utf8") as ranking_file:
    for search_ranking in search_rankings:
        ranking_file.write(json.dumps(search_ranking) + "\n")

# Save the log
log_file_path = output_path + "evaluation-log-single-g" + str(GAMMA) + "-l" + str(LAMBDA) + "-b" + \
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
