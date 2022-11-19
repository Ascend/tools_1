# Profiling Data Analysis version 2
import os
from collections import defaultdict
import re
import importlib
import importlib.util
from .definitions import *
from . import reporter_registry

DEVICE_EVENT = "[Compute]"


class ProfilingDataAnalyzer:
    V1_START_RE = re.compile(r'Profiler version: (?P<version>[\d+.]+), dump start, records num: \d+')
    V1_RE = re.compile(
        r'(?P<timestamp>\d+) (?P<tid>\d+) (?P<element>(\[\S+\])|(UNKNOWN\(-?\d+\))) (?P<event>(\[\S+\])|(UNKNOWN\(-?\d+\))) (?P<et>(Start)|(End)|(UNKNOWN\(-?\d+\)))')

    def __init__(self, file_path):
        self._file_path = file_path

    def read_in_records(self):
        pds = []
        with open(self._file_path, 'r') as f:
            pd = ProfilingData()
            for line in f:
                ma = ProfilingDataAnalyzer.V1_START_RE.match(line)
                if ma is not None:
                    pd.version = ma.group('version')
                    continue
                if line.startswith("Profiling dump end"):
                    pds.append(pd)
                    pd = ProfilingData()
                    continue
                ma = ProfilingDataAnalyzer.V1_RE.match(line)
                if ma is not None:
                    rec = Record()
                    rec.timestamp = int(ma.group("timestamp"))
                    rec.tid = int(ma.group("tid"))
                    rec.node_name = ma.group("element")
                    if rec.node_name.startswith("UNKNOWN"):
                        rec.node_name = None
                    rec.event = ma.group("event")
                    if rec.event == DEVICE_EVENT:
                        rec.device = DEV_DEVICE
                        rec.tid = 1
                    rec.et = ma.group("et")
                    pd.add_record(rec)
                    continue
                # logging.warning("Skip unrecognized line {}".format(line))
        return pds

    @staticmethod
    def read_in_event_records(pds: [ProfilingData]):
        for pd in pds:
            records = pd.records[:]
            records.sort(key=lambda tmp_rec: tmp_rec.timestamp)
            keys_to_starts = defaultdict(list)
            for rec in records:
                name = get_name(rec)
                if rec.et == 'Start':
                    keys_to_starts[name].append(rec)
                    continue
                if rec.et == 'End':
                    if len(keys_to_starts[name]) == 0:
                        print("WARNING: Drop record {}, because can not find a start record for it".format(rec))
                        continue
                    start_rec = keys_to_starts[name].pop()
                    er = EventRecord.create_from_rec(start_rec)
                    er.start = start_rec.timestamp
                    er.end = rec.timestamp
                    pd.add_event_record(er)
            for starts in keys_to_starts.values():
                for no_end_start_rec in starts:
                    print("WARNING: Drop record {}, because can not find a end event for it".format(no_end_start_rec))

    def read_in_profiling_file(self):
        pds = self.read_in_records()
        ProfilingDataAnalyzer.read_in_event_records(pds)
        return pds


def load_all_builtin_reporters():
    ada_path = os.path.dirname(__file__)
    builtin_reporters_path = os.path.join(ada_path, "builtin_reporters")
    for file_name in os.listdir(builtin_reporters_path):
        if not file_name.endswith("_reporter.py"):
            continue
        file_path = os.path.join(builtin_reporters_path, file_name)
        if not os.path.isfile(file_path):
            continue
        module_name = file_name[:-3]
        if importlib.util.find_spec(module_name) is not None:
            continue
        importlib.util.spec_from_file_location(module_name, file_path).loader.load_module(module_name)


def get_all_reporters():
    load_all_builtin_reporters()
    return reporter_registry.get_all_reporter_names()


class ReporterBuilder:
    def __init__(self, name, builder):
        self.name = name
        self.builder = builder
        self.error_message = None

    @staticmethod
    def duplicate(name):
        builder = ReporterBuilder(name, None)
        builder.error_message = "The reporter has been called"


class Reporters:
    def __init__(self):
        self.categories_to_reporter_builders = defaultdict(list)

    def add_reporter(self, category, reporter_builder):
        self.categories_to_reporter_builders[category].append(reporter_builder)

    def only_dump_basic(self):
        return "basic" in self.categories_to_reporter_builders and len(self.categories_to_reporter_builders) == 1

    @staticmethod
    def get_names_from_type(reporter_type):
        names = reporter_registry.get_names_by_category(reporter_type)
        if len(names) > 0:
            return names
        if reporter_registry.get_reporter(reporter_type) is not None:
            return [reporter_type, ]
        return []

    @staticmethod
    def create_from_types(types):
        reporters = Reporters()
        if types is None:
            types = ["basic", ]

        found_names = set()
        not_found_categories = set()
        for report_type in types:
            names = Reporters.get_names_from_type(report_type)
            if len(names) == 0:
                not_found_categories.add(report_type)
                continue

            for name in names:
                if name in found_names:
                    reporters.add_reporter(report_type, ReporterBuilder.duplicate(name))
                else:
                    reporters.add_reporter(report_type, ReporterBuilder(name, reporter_registry.get_reporter(name)))

        # todo not support external reporters, print error when reporters not found
        if len(not_found_categories) != 0:
            raise AdaError("Reporters not found: {}".format(','.join(not_found_categories)))
        return reporters


def main_ge(args):
    try:
        path, tracing_path, reporter_categories = args.input_file, args.output, args.reporter
        analyzer = ProfilingDataAnalyzer(path)
        pds = analyzer.read_in_profiling_file()

        load_all_builtin_reporters()
        reporters = Reporters.create_from_types(reporter_categories)
        for category in reporters.categories_to_reporter_builders:
            indent = ''
            if not reporters.only_dump_basic():
                print("Dump {}:".format(category))
                indent = '  '
            for reporter_builder in reporters.categories_to_reporter_builders[category]:
                if reporter_builder.builder is None:
                    print("{}{}".format(indent, reporter_builder.message))
                else:
                    print("{}Begin to generate report {}...".format(indent, reporter_builder.name), end='')
                    reporter = reporter_builder.builder(pds)
                    reporter.report(tracing_path)
                    print("done")
        return 0
    except AdaError as e:
        print("Error: {}".format(e.message))
        return 1
