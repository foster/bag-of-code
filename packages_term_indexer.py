from collections import OrderedDict
from scipy import linalg
import math
import numpy as np

class PackagesTermIndexer:
    def __init__(self):
        self._packages = []
        self.__reset()
        self._global_term_count = OrderedDict()

    # reset all internal state.
    # used every time fit() is called.
    def __reset(self):
        self.index = None
        self._global_weights_ = None
        self._word_frequency_matrix_ = None
        self._tfidf_matrix_ = None
        self._svd_ = None

    # fills the indexer with the set of all packages to train on
    # resets all internal state, calling this repeatedly is destructive
    def append(self, package):
        self.__reset()
        self._packages.append(package)

        # build corpus (all terms => global count of occurance)
        for term, local_count in package.iteritems():
            self.__increment_global_term_count(term, local_count)

    # fit() calls this method to incrementally build a list of all terms
    # and record how many times each term occured
    def __increment_global_term_count(self, term, count=1):
        # initialize term count if necessary
        if term not in self._global_term_count:
            self._global_term_count[term] = 0

        # increment count for this term
        self._global_term_count[term] += count

    @property
    def term_indices_(self):
        return self._global_term_count.keys()
    @property
    def package_names_(self):
        return map(lambda p: p.name, self._packages)

    # calculate term frequency-inverse document frequency for the dataset
    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    @property
    def global_weights_(self):
        # memoize the result
        if self._global_weights_ is not None:
            return self._global_weights_
        self._global_weights_ = {}

        # logn: log base 2 of the number of packages
        logn = math.log(len(self._packages), 2)

        for term, global_count in self._global_term_count.iteritems():
            weight = 1.0

            for pkg in self._packages:
                if term in pkg:
                    local_count = pkg[term]
                    l_to_g_ratio = 1.0 * local_count / global_count
                    weight += l_to_g_ratio * math.log(l_to_g_ratio, 2) / logn

            self._global_weights_[term] = weight
        return self._global_weights_

    # word frequency matrix
    # this is a matrix where every row represents a package
    # and every column represents a term in the corpus
    # the cells represent how many times any term appears in the package
    def word_frequency_matrix(self):
        # memoize the result
        if self._word_frequency_matrix_ is not None:
            return self._word_frequency_matrix_

        # build a term => index lookup
        all_terms = self._global_term_count.keys()
        index = dict([(val, idx) for idx, val in enumerate(all_terms)])

        pkg_vecs = []
        for pkg in self._packages:
            # initialize row to be a bunch of zeros
            row = [0] * len(self._global_term_count)

            # fill in the frequency for terms that exist in this package
            for term, local_count in pkg.iteritems():
                idx = index[term]
                row[idx] = local_count
            pkg_vecs.append(row)

        self._word_frequency_matrix_ = np.matrix(pkg_vecs).transpose()
        return self._word_frequency_matrix_

    # term frequency-inverse document frequency matrix
    # this is similar to a word frequency matrix, but the values are
    # weighted by how (in)frequently they appear in the corpus (all indexed packages)
    def tfidf_matrix(self):
        # memoize the result
        if self._tfidf_matrix_ is not None:
            return self._tfidf_matrix_

        global_weights_by_index = self.global_weights_.values()

        def wf_to_tfidf(idx, local_count):
            global_weight = global_weights_by_index[idx]
            ln_local_count = math.log(local_count + 1.0, 2)
            return global_weight * ln_local_count
        def wf_vec_to_tfidf_vec(vector):
            return [wf_to_tfidf(i,v) for i,v in enumerate(vector)]

        wfm = self.word_frequency_matrix()
        self._tfidf_matrix_ = np.apply_along_axis(wf_vec_to_tfidf_vec, 1, wfm)
        return self._tfidf_matrix_

    def svd(self):
        # memoize the result
        if self._svd_ is not None:
            return self._svd_
        T, sigma, D_trans = linalg.svd(self.tfidf_matrix(), full_matrices=False)
        self._svd_ = (T, sigma, D_trans)
        return self._svd_

    # map an unknown package to our term space
    # result is a word frequency matrix
    def fold_wfm(self, package):
        vec = [0] * len(self._global_term_count)
        for idx, term in enumerate(self._global_term_count):
            if term not in package: continue
            vec[idx] = package[term]
        return vec

    # map an unknown package to our term space
    # result is a tfidf matrix
    def fold_tfidf(self, package):
        vec = [0] * len(self._global_term_count)
        for idx, term in enumerate(self._global_term_count):
            if term not in package: continue
            global_weight = self.global_weights_[term]
            local_count = package[term]
            ln_local_count = math.log(local_count + 1.0, 2)
            vec[idx] = global_weight * ln_local_count
        return vec

    # map an unknown package to our term space
    # result is an svd
    def fold_svd(self, package):
        T, sigma, D_trans = self.svd()
        pkg_vector = self.fold_tfidf(package)

        n = D_trans.shape[1]
        sigma_inv = np.linalg.inv(linalg.diagsvd(sigma, n, n))
        vec = np.matrix(pkg_vector)
        folded_vec = np.dot(np.dot(vec, T), sigma_inv)
        return folded_vec
        


class Package:
    def __init__(self, name):
        self.name = name
        self._local_term_count = {}

    def keys(self):
        return self._local_term_count.keys()

    def __getitem__(self, key):
        return self._local_term_count[key]

    def __contains__(self, key):
        return key in self._local_term_count

    def iteritems(self):
        return self._local_term_count.iteritems()

    def register_term(self, term):        
        # initialize term count if necessary
        if term not in self._local_term_count:
            self._local_term_count[term] = 0

        self._local_term_count[term] += 1
