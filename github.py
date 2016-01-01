import json
import numpy as np
import requests
from os.path import isfile

LANG_CACHE_FILE_NAME = '.cache/repo-languages.json'

class Github:
    def __init__(self, github_user=None, github_token=None):
        if github_user and github_token:
            self._github_api_auth = (github_user, github_token)
        else:
            self._github_api_auth = None

    def is_repo_javascript(self, gh_url, verbose=False):
        all_repo_langs = {}
        this_repo_langs = {}
        if isfile(LANG_CACHE_FILE_NAME):
            with open(LANG_CACHE_FILE_NAME, 'r') as f:
                all_repo_langs = json.load(f)

        if not gh_url in all_repo_langs:
            all_repo_langs[gh_url] = self._lookup_languages_on_github_api(gh_url)
            with open(LANG_CACHE_FILE_NAME, 'w') as f:
                json.dump(all_repo_langs, f)

        this_repo_langs = all_repo_langs.get(gh_url) or {}
        js_lines = this_repo_langs.get('JavaScript', 0)
        total_lines = np.sum(this_repo_langs.values(), dtype=np.int32)
        total_lines = total_lines if total_lines > 0 else 1
        percent_js = 100.0 * js_lines / total_lines
        if verbose:
            print "{} javascript: {} / {} ({:.1f})".format(gh_url, js_lines, total_lines, percent_js)
        return percent_js > 80

    def _lookup_languages_on_github_api(self, gh_url):
        languages_url = 'https://api.github.com/repos/{}/languages'.format(gh_url)

        resp = requests.get(languages_url, auth=self._github_api_auth)
        if resp.status_code == requests.codes.not_found:
            print 'Not found: {}'.format(gh_url)
            return None
        elif resp.status_code == requests.codes.forbidden and resp.headers.get('X-RateLimit-Remaining') is '0':
                raise RuntimeError('Github rate limit exceeded. Are you authenticated?')
        elif resp.status_code is not requests.codes.ok:
            resp.raise_for_status()

        languages = resp.json()
        # delete HTML and CSS. They're documentation and shouldn't count toward % of code
        languages.pop('HTML', None)
        languages.pop('CSS', None)
        return languages
