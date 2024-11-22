import itertools

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from mbmbm import logger


class IndValSelector(SelectorMixin, BaseEstimator):
    def __init__(
        self,
        percentile=50,
        indval_func="indval.g",
        num_permutations=20,
        min_order=1,
        max_order=None,
    ) -> None:
        super().__init__()

        self.percentile = percentile
        self.indval_func = indval_func
        self.num_permutations = num_permutations
        self.min_order = min_order
        self.max_order = max_order

    def fit(self, X, y):
        # Ensure y is a string for categorical handling
        y = y.astype(str)

        # Extract species names and cluster names
        species_names = np.arange(X.shape[1])
        cluster_names, y_indices = np.unique(y, return_inverse=True)

        logger.debug(f"{cluster_names=}")

        cluster_number = len(cluster_names)

        # Determine max order
        if self.max_order:
            max_order = self.max_order
        else:
            max_order = cluster_number

        # Generate possible combinations of clusters
        combinations, combination_names = IndValSelector.create_cluster_combinations(
            cluster_number, self.min_order, max_order
        )

        # Build the plot membership matrix corresponding to combinations
        membership = combinations[y_indices]

        # Original Membership matrix with only single clusters
        membership_original = membership[:, :cluster_number]

        # Compute association strength for each group
        if self.indval_func == "indval":
            mode = "site"
        elif self.indval_func == "indval.g":
            mode = "group"
        else:
            raise NotImplementedError(f"Unknown func: {self.indval_func}")

        self.indval_result = IndValSelector.indvalcomb(
            X, membership_original, membership, self.min_order, max_order, mode=mode
        )

        # Extract values from indval result
        stat = self.indval_result["iv"]  # Indicator value
        A = self.indval_result["A"]  # Specificity
        B = self.indval_result["B"]  # Fidelity

        # Maximum association strength
        maxstr = stat.max(axis=1)

        # Occurrence in combinations
        wmax = stat.argmax(axis=1)

        # Prepare results
        m = combinations[:, wmax].T
        combination_indices = np.array([combination_names.index(a) + 1 for a in wmax])
        m = np.column_stack((m, combination_indices, maxstr))

        # Perform permutations and compute p-values
        pv = np.ones(len(species_names))
        for _ in range(self.num_permutations):
            tmpclind = np.random.permutation(y_indices)
            combp = combinations[tmpclind]
            membp = combp[:, :cluster_number]

            indval_result_p = IndValSelector.indvalcomb(X, membp, combp, self.min_order, max_order, mode=mode)
            stat_p = indval_result_p["iv"]

            tmpmaxstr = stat_p.max(axis=1)
            pv += (tmpmaxstr > maxstr).astype(int)

        pvalues = pv / (1 + self.num_permutations)

        # Adjust p-values for criteria
        criterium = m[:, -2] == (2**cluster_number - 1)
        pvalues[criterium] = np.nan
        pvalues[~criterium] = pvalues[~criterium]

        # Collect results
        self.indval_result = (stat, m, A, B)
        self.indicator_values = m[:, -1]

        return self

    def _get_support_mask(self):
        return self.indicator_values > np.nanpercentile(self.indicator_values, self.percentile)

    @staticmethod
    def create_cluster_combinations(k, min_order, max_order):
        epn_matrix = []
        epn_names = []
        for j in range(max(min_order, 1), min(max_order, k) + 1):
            for combination in itertools.combinations(range(k), j):
                this_list = np.zeros(k, dtype=int)
                for c in combination:
                    this_list[c] = 1
                epn_matrix.append(this_list)
                epn_names.append(combination)

        epn_matrix = np.array(epn_matrix).T

        return epn_matrix, epn_names

    @staticmethod
    def indvalcomb(x, memb, comb, min_order, max_order, mode="group"):
        k = memb.shape[1]
        t_x = x.T

        aisp = t_x @ comb
        ni = comb.sum(axis=0)
        nisp = (t_x > 0) @ comb

        nispni = nisp.astype(float) / ni

        if mode == "site":
            A = aisp.astype(float)
            for i in range(x.shape[1]):
                A[i, :] /= x[:, i].sum()
        else:
            aispni = aisp.astype(float) / comb.sum(axis=0)
            asp = aispni[:, :3].sum(axis=1)

            if max_order == 1:
                A = aispni / asp[:, None]
            else:
                A = np.zeros_like(aispni)
                for j in range(min_order, max_order + 1):
                    for combination in itertools.combinations(range(k), j):
                        indices = list(combination)
                        A[:, indices] = aispni[:, indices].sum(axis=1, keepdims=True) / asp[:, None]

        iv = np.sqrt(A * nispni)

        return {"A": A, "B": nispni, "iv": iv}
