from collections import defaultdict
import os
import inspect


_reporters = {}
_categories_to_reporter_names = defaultdict(list)
_files_to_reporter_names = defaultdict(list)


def get_all_reporter_names():
    return _reporters.keys()


def get_names_by_category(category):
    return _categories_to_reporter_names.get(category, [])


def get_reporter(name):
    return _reporters[name]


def reporter(name, category=None):
    def register_reporter(cls):
        if name in _reporters:
            print("Duplicate report name {}".format(name))
            return
        _reporters[name] = cls
        if category is None:
            _files_to_reporter_names[os.path.realpath(inspect.getfile(cls))].append(name)
        else:
            _categories_to_reporter_names[category].append(name)
        _categories_to_reporter_names["all"].append(name)
    return register_reporter
