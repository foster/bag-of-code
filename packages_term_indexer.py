from collections import OrderedDict
import math

class PackagesTermIndexer:
    def __init__(self):
        self._global_term_count = OrderedDict()
        self._packages = []

    def __getitem__(self, key):
        return self._global_term_count[key]

    def keys(self):
        return self._global_term_count.keys()

    def iteritems(self):
        return self._global_term_count.iteritems()

    def increment_global_term_count_(self, term):
        self._global_weights_ = None

        # initialize term count if necessary
        if term not in self._global_term_count:
            self._global_term_count[term] = 0

        # increment count for this term
        self._global_term_count[term] += 1

    def register_package(self, name):
        self._global_weights_ = None
        pkg = _Package(self, name)
        self._packages.append(pkg)
        return pkg

    @property
    def term_indices(self):
        return self.keys()

    @property
    def package_names(self):
        return map(lambda p: p.name, self._packages)

    # calculate term frequency-inverse document frequency for the dataset
    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    @property
    def global_weights(self):
        # memoize the result
        if self._global_weights_:
            return self._global_weights_
        self._global_weights_ = {}

        # logn: log base 2 of the number of packages
        logn = math.log(len(self._packages), 2)

        for term, global_count in self.iteritems():
            weight = 1.0

            for pkg in self._packages:
                if term in pkg:
                    local_count = pkg[term]
                    l_to_g_ratio = 1.0 * local_count / global_count
                    weight += l_to_g_ratio * math.log(l_to_g_ratio, 2) / logn

            self._global_weights_[term] = weight
        return self._global_weights_

    def word_frequency_matrix(self):
        # build a term => index lookup
        index = dict([(val, idx) for idx, val in enumerate(self.keys())])

        tdm = []
        for pkg in self._packages:
            # initialize row to be a bunch of zeros
            row = [0] * len(self._global_term_count)

            # fill in the frequency for terms that exist in this package
            for term, local_count in pkg.iteritems():
                idx = index[term]
                row[idx] = local_count
            tdm.append(row)
        return tdm

    # term frequency-inverse document frequency matrix
    # this is similar to a word_frequency_matrix, but the values are
    # weighted by how (in)frequently they appear in the corpus (all indexed packages)
    def tfidf_matrix(self):
        global_weights_by_index = self.global_weights.values()

        def wf_to_tfidf(idx, local_count):
            global_weight = global_weights_by_index[idx]
            ln_local_count = math.log(local_count + 1.0, 2)
            return global_weight * ln_local_count

        wf_matrix = self.word_frequency_matrix()
        return map(lambda row: [wf_to_tfidf(i,v) for i,v in enumerate(row)], wf_matrix)

class _Package:
    def __init__(self, parent, name):
        self.name = name
        self._parent = parent
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
        self._parent.increment_global_term_count_(term)
        
        # initialize term count if necessary
        if term not in self._local_term_count:
            self._local_term_count[term] = 0

        self._local_term_count[term] += 1
