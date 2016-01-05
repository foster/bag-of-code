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

    def tdm(self):
        # build a term => index lookup
        index = dict([(val, idx) for idx, val in enumerate(self.keys())])

        tdm = []
        for pkg in self._packages:
            # initialize row to be a bunch of zeros
            row = [0] * len(self._global_term_count)

            # fill in the weight for terms that do exist in this package
            for term, local_count in pkg.iteritems():
                idx = index[term]
                lg_term = math.log(local_count + 1.0, 2)
                row[idx] = self.global_weights[term] * lg_term
            tdm.append(row)
        return tdm


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
