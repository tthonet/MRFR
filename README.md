**Multi-grouping Robust Fair Ranking**
======

## __Introduction__

This repository provides the source code for *Multi-grouping Robust Fair Ranking* by Thibaut Thonet and Jean-Michel Renders. The implementation is based on Python. More details about this work can be found in the original paper, which is available at https://dl.acm.org/doi/abs/10.1145/3397271.3401292.

**Abstract:** Rankings are at the core of countless modern applications and thus play a major role in various decision making scenarios. When such rankings are produced by data-informed, machine learning-based algorithms, the potentially harmful biases contained in the data and algorithms are likely to be reproduced and even exacerbated. This motivated recent research to investigate a methodology for fair ranking, as a way to correct the aforementioned biases. Current approaches to fair ranking consider that the protected groups, i.e., the partition of the population potentially impacted by the biases, are known. However, in a realistic scenario, this assumption might not hold as different biases may lead to different partitioning into protected groups. Only accounting for one such partition (i.e., grouping) would still lead to potential unfairness with respect to the other possible groupings. Therefore, in this paper, we study the problem of designing fair ranking algorithms without knowing in advance the groupings that will be used later to assess their fairness. The approach that we follow is to rely on a carefully chosen set of groupings when deriving the ranked lists, and we empirically investigate which selection strategies are the most effective. An efficient two-step greedy brute-force method is also proposed to embed our strategy. As benchmark for this study, we adopted the dataset and setting composing the TREC 2019 Fair Ranking track.

If you found this implementation useful, please consider citing us:

Thonet, T., & Renders, J.-M. (2020). **[Multi-grouping Robust Fair Ranking](https://dl.acm.org/doi/abs/10.1145/3397271.3401292)**. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2077-2080.

## __Content__

The repository contains the following files:

* The directory **data** containing the data used for the experiments (coming from the [TREC Fair Ranking 2019 track](https://fair-trec.github.io/2019/)) as well as the source and target grouping definition files.
* The directory **evaluation** containing the scripts to evaluate the model. These correspond to a slightly modified version of the evaluation scripts released by the organizers of [TREC Fair Ranking 2019](https://fair-trec.github.io/2019/).
* The directory **models** containing the scripts to run the Single-query Greedy Brute-force Re-ranking (SGBR) approach.
* The file **LICENCE.txt** describing the licence of our code.
* The file **README.md**, which is the current file.

## __How to run the code__

The Single-query Greedy Brute-force Re-ranking (SGBR) approach can be run using two scripts, depending on the type of source groupings considered: ``sgbr_single.py`` and ``sgbr_aggreg.py``. The first one corresponds to the case where only a single grouping is considered for the definition of fairness, while the second one corresponds to the multi-grouping case. They are run using the same arguments as follows:

``sgbr_single.py [-h] -f <string> [-g <float>] [-l <float>] [-t <float>] [-e <float>] [-b <int>] [-n <int>]``
``sgbr_aggreg.py [-h] -f <string> [-g <float>] [-l <float>] [-t <float>] [-e <float>] [-b <int>] [-n <int>]``

The meaning of each argument is detailed below:

* ``-h``, ``--help``: Show usage.
* ``-f <string>``, ``--grouping_file <string>``: Name of the file(s) containing the source grouping which defines the fairness used to derive the rankings. In the case of ``sgbr_single.py`` the string will contain only one file name, while in the case of ``sgbr_aggreg.py`` the string will contain a sequence of file names (each corresponding to a grouping), separated by spaces.
* ``-g <float>``, ``--gamma <float>``: Continuation probability in the exposure model. Default value: 0.9.
* ``-l <float>``, ``--lambd <float>``: Weight of fairness with respect to utility in the optimization objective. Default value: 1.0.
* ``-t <float>``, ``--beta <float>``: Weight of deserved exposure with respect to relevance in the item pre-ordering phase. Default value: 1.0.
* ``-e <float>``, ``--eps <float>``: Probability of satisfaction given a relevant document. Default value: 0.5.
* ``-b <int>``, ``--bin_size <int>``: Size of the bins of documents. A bin corresponds to a group of consecutive documents in a given ranking, and ranking candidates are formed by permuting documents only within bins. This corresponds to the hyperparameter K in the paper. Default value: 3.
* ``-n <int>``, ``--max_n_bins <int>``: Maximum number of bins to split rankings into. Default value: 1.

When run, these scripts generate two files in ``data/results/<grouping_file argument>``: a ``.txt`` log file and a ``.json`` file corresponding to the rankings formed for the input query sequences.

## __Evaluation__

The rankings generated by the SGBR approach are evaluated using the scripts in the ``evaluation`` directory. The script ``trec-fair-ranking-evaluator.py`` corresponds to a slightly modified version of the official [TREC Fair Ranking 2019](https://fair-trec.github.io/2019/) evaluation script. The evaluation on the target groupings is performed by (1) copying the SGBR-generated .json ranking file(s) in ``evaluation\fairRuns``; (2) adding the name of those ranking files (minus the .json extension) at line 233 of ``trec-fair-ranking-evaluator.py``; (3) running ``run_eval.sh`` on the preferred set of target groupings (either C4T or C8T).
