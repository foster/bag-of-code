from collections import OrderedDict
from scipy import linalg
import math
import numpy as np

class PackagesTermIndexer:
    def __init__(self):
        self._packages = []
        self._global_term_count = OrderedDict()

    # fills the indexer with the set of all packages to train on
    # resets all internal state, calling this repeatedly is destructive
    def append(self, package):
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

    def fit_trim(self):
        return LSIDoer(self._packages, self._global_term_count, feature_limit=200)


class Package:
    def __init__(self, name, github_url):
        self.name = name
        self.github_url = github_url
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


class LSIDoer:
    def __init__(self, packages, global_term_count, feature_limit=1000):
        self._global_term_count = OrderedDict(sorted(global_term_count.iteritems(), key=lambda x: x[1], reverse=True)[0:feature_limit])
        self._packages = packages
        self._global_weights_ = None
        self._word_frequency_matrix_ = None
        self._tfidf_matrix_ = None
        self._svd_ = None
        self._svd_wfm_ = None
        self._package_vitals_ = [(p.name, p.github_url) for p in packages]

    # provide custom method to pickle this object
    # with as little information as necessary
    def __getstate__(self):
        return {
            '_global_weights_': self.global_weights_,
            '_word_frequency_matrix_': self.word_frequency_matrix(),
            '_tfidf_matrix_': self.tfidf_matrix(),
            '_svd_': self.svd(),
            '_svd_wfm_': self.svd_wfm(),
            '_package_vitals_': self._package_vitals_
        }

    @property
    def term_indices_(self):
        return self.global_weights_.keys()

    @property
    def package_names_(self):
        return [p[0] for p in self._package_vitals_]

    @property
    def package_github_urls_(self):
        return [p[1] for p in self._package_vitals_]


    # calculate term frequency-inverse document frequency for the dataset
    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    @property
    def global_weights_(self):
        # memoize the result
        if self._global_weights_ is not None:
            return self._global_weights_
        self._global_weights_ = OrderedDict()

        # logn: log base 2 of the number of packages
        logn = math.log(len(self._packages), 2)

        for term, global_count in self._global_term_count.iteritems():
            weight = 1.0

            for pkg in self._packages:
                if term in pkg:
                    local_count = pkg[term]
                    l_to_g_ratio = 1.0 * local_count / global_count
                    weight += (l_to_g_ratio * math.log(l_to_g_ratio, 2) / logn)

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
        all_terms = self.global_weights_.keys()
        index = dict([(val, idx) for idx, val in enumerate(all_terms)])

        pkg_vecs = []
        for pkg in self._packages:
            # initialize row to be a bunch of zeros
            row = [0] * len(self.global_weights_)

            # fill in the frequency for terms that exist in this package
            for term, local_count in pkg.iteritems():
                if term not in index: continue
                idx = index[term]
                row[idx] = local_count
            pkg_vecs.append(row)

        self._word_frequency_matrix_ = np.matrix(pkg_vecs).transpose()
        return self._word_frequency_matrix_

    # term frequency-inverse document frequency matrix
    # this is similar to a word frequency matrix, but the values are
    # weighted by how (in)frequently they appear in the corpus (all indexed packages)
    def tfidf_matrix(self):
        # implementation note: it appears faster to build a new matrix from scratch
        # than try to map/transform a word frequency matrix
        global_weights_by_index = self.global_weights_.values()

        # build a term => index lookup
        all_terms = self.global_weights_.keys()
        index = dict([(val, idx) for idx, val in enumerate(all_terms)])
        row_len = len(index)

        pkg_rows = []
        for pkg in self._packages:
            # initialize row to be a bunch of zeros
            row = [0] * row_len

            # fill in the frequency for terms that exist in this package
            for term, local_count in pkg.iteritems():
                if term not in index: continue
                idx = index[term]
                global_weight = global_weights_by_index[idx]

                ln_local_count = math.log(local_count + 1.0, 2)
                row[idx] = global_weight * ln_local_count

            pkg_rows.append(row)

        return np.matrix(pkg_rows).transpose()

    def svd(self):
        # memoize the result
        if self._svd_ is not None:
            return self._svd_
        T, sigma, D_trans = linalg.svd(self.tfidf_matrix(), full_matrices=False)
        self._svd_ = (T, sigma, D_trans)
        return self._svd_

    def svd_wfm(self):
        #memoize the result
        if self._svd_wfm_ is not None:
            return self._svd_wfm_
        T, sigma, D_trans = linalg.svd(self.word_frequency_matrix(), full_matrices=False)
        self._svd_wfm_ = (T, sigma, D_trans)
        return self._svd_wfm_

    # map an unknown package to our term space
    # result is a word frequency matrix
    def fold_wfm(self, package):
        vec = [0] * len(self.global_weights_)
        for idx, term in enumerate(self.global_weights_):
            if term not in package: continue
            vec[idx] = package[term]
        return np.matrix(vec)

    # map an unknown package to our term space
    # result is a tfidf matrix
    def fold_tfidf(self, package):
        vec = [0] * len(self.global_weights_)
        for idx, term in enumerate(self.global_weights_):
            if term not in package: continue
            global_weight = self.global_weights_[term]
            local_count = package[term]
            ln_local_count = math.log(local_count + 1.0, 2)
            vec[idx] = global_weight * ln_local_count
        return np.matrix(vec)

    # map an unknown package to our term space
    # result is an svd
    def fold_svd(self, package):
        # T is a matrix representing the terms.
        # D is a matrix representing the documents.
        # sigma is a diagonal matrix of singular values.
        T, sigma, D_t = self.svd()

        sigma_full_matrix = linalg.diagsvd(sigma, len(sigma), len(sigma))
        sigma_inv = np.linalg.inv(sigma_full_matrix)

        folded_tfidf = self.fold_tfidf(package)

        # calculate a representation of the new document as a vector of SVDs
        folded_doc_dot_T = np.dot(folded_tfidf, T)
        folded_svd_vec = np.dot(folded_doc_dot_T, sigma_inv)

        return folded_svd_vec

    def fold_svd_wfm(self, package):
        # T is a matrix representing the terms.
        # D is a matrix representing the documents.
        # sigma is a diagonal matrix of singular values.
        T, sigma, D_t = self.svd()

        sigma_full_matrix = linalg.diagsvd(sigma, len(sigma), len(sigma))
        sigma_inv = np.linalg.inv(sigma_full_matrix)

        folded_wfm = self.fold_wfm(package)

        # calculate a representation of the new document as a vector of SVDs
        folded_doc_dot_T = np.dot(folded_wfm, T)
        folded_svd_vec = np.dot(folded_doc_dot_T, sigma_inv)

        return folded_svd_vec
