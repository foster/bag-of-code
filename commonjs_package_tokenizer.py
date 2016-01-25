from os.path import isfile, islink
import codecs
import json
import os
import re
import shlex

def tokenize_package(dir_name):
    path = os.path.join('.cache/packages', dir_name)
    path_to_package_json = os.path.join(path, 'package.json')

    if not isfile(path_to_package_json):
        return

    lib_dir = _get_lib_dir(path_to_package_json)
    lib_path = os.path.join(path, lib_dir) if lib_dir else path

    for dirpath, dirnames, filenames in os.walk(lib_path):
        forbidden_dirs = [
            '/.git', '/docs', '/benchmark', '/benchmarks', '/i18n', '/images',
            '/test', '/tests', '/examples', '/tutorials', '/vendor', '/node_modules'
        ]
        if reduce(lambda default, forbidden_d: 1 if forbidden_d in dirpath else default, forbidden_dirs, 0):
            continue

        for filename in filenames:
            if not filename.endswith('.js'):
                continue
            file_path = os.path.join(dirpath, filename)
            if islink(file_path):
                continue

            with codecs.open(file_path, 'r', encoding='utf8') as f:
                try:
                    for line in f:
                        tokens = shlex.split(line, comments=['#', '//'])
                        for t in tokens:
                            yield t
                except ValueError, UnicodeDecodeError:
                    pass

def _get_lib_dir(path_to_package_json):
    package_json_file = open(path_to_package_json)
    package_json = json.load(package_json_file)
    package_json_file.close()

    lib_dir = package_json.get('directories', {}).get('lib', '')
    # filter for packages that don't adhere to CommonJS spec and start with dot-slash:
    lib_dir = re.sub('^\./', '', lib_dir)
