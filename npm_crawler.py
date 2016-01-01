import re
import json
from bs4 import BeautifulSoup
from os.path import isfile
from urllib import urlretrieve, urlopen

MOST_INSTALLED_DATA_URL = 'https://raw.githubusercontent.com/npm/npm-explicit-installs/master/data.json'
MOST_INSTALLED_DATA_FILE_PATH = '.cache/npm-explicit-installs-data.json'
NPM_MOST_STARRED_HTML_URL = 'https://www.npmjs.com/browse/star'
NPM_MOST_STARRED_HTML_FILE_PATH = '.cache/npm-most-installed.html'
NPM_REGISTRY_DATA_URL = 'http://registry.npmjs.org/{}/latest'


def get_most_installed_packages():
    if not isfile(MOST_INSTALLED_DATA_FILE_PATH):
        urlretrieve(MOST_INSTALLED_DATA_URL, MOST_INSTALLED_DATA_FILE_PATH)
        
    data_file = open(MOST_INSTALLED_DATA_FILE_PATH)
    data = json.load(data_file)
    data_file.close()

    for pkg in data:
        name = pkg.get('name')
        repo_url = pkg.get('repository', {}).get('url')
        repo_github = _parse_github_short_url(repo_url)
        if repo_github:
            yield (name, repo_github)


def get_most_starred_packages(max_pages=1):
    for package_name in _get_most_starred_package_names(max_pages):
        file_name = '.cache/npm-registry-' + package_name + '.json'
        if not isfile(file_name):
            url = NPM_REGISTRY_DATA_URL.format(package_name)
            urlretrieve(url, file_name)

        data_file = open(file_name)
        data = json.load(data_file)
        data_file.close()
        name = data.get('name')
        repo_url = data.get('repository', {}).get('url')
        repo_github = _parse_github_short_url(repo_url)
        if repo_github:
            yield (name, repo_github)


def _get_most_starred_package_names(max_depth=1, depth=1):
    next_url = NPM_MOST_STARRED_HTML_URL

    while depth <= max_depth:
        file_name = NPM_MOST_STARRED_HTML_FILE_PATH + '.' + str(depth)
        if not isfile(file_name):
            urlretrieve(next_url, file_name)

        html_file = open(file_name)
        soup = BeautifulSoup(html_file)
        html_file.close()

        next_buttons = soup.select('.pagination a.next')
        elements = soup.select('.package-details a.name')
        for e in elements:
            yield e.string
    
        if len(next_buttons) > 0:
            next_button = next_buttons[0]
            next_url = 'https://www.npmjs.com' + next_button['href']
            depth += 1
        else:
            break

def _parse_github_short_url(repo_url):
    regex = 'github.com/(\S+?/\S+?)\.git'
    match = re.search(regex, repo_url) if repo_url else None
    if match:
        return match.group(1)
    else:
        return None

