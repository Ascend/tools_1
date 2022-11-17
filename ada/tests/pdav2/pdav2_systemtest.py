import os.path
import unittest
from unittest.mock import patch
import sys
import json
import shutil
import ada_prof_cmd


class AdaPaSystemTest(unittest.TestCase):
    log_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ada_perf_data"))
    result_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp-output"))

    def tearDown(self) -> None:
        super().tearDown()
        AdaPaSystemTest.clear()

    @staticmethod
    def clear():
        for file_name in os.listdir(AdaPaSystemTest.result_dir):
            if file_name == ".gitkeep":
                continue
            file_path = os.path.join(AdaPaSystemTest.result_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(os.path.join(AdaPaSystemTest.result_dir, file_name))

    def assert_show_in_ns(self, result_file_path):
        with open(result_file_path, "r") as f:
            jf = json.load(f)
        self.assertTrue("displayTimeUnit" in jf)
        self.assertEqual(jf["displayTimeUnit"], "ns")

    def assert_records_num(self, file_path, num):
        with open(file_path, "r") as f:
            jf = json.load(f)
        self.assertTrue("traceEvents" in jf)
        self.assertEqual(len(jf["traceEvents"]), num)

    def test_default_reports(self):
        with patch.object(sys, 'argv', ["ada-pa",
                                        os.path.join(AdaPaSystemTest.log_dir, "pytorch_ns.log"),
                                        "--output={}".format(AdaPaSystemTest.result_dir)]):
            self.assertEqual(ada_prof_cmd.main(), 0)
        self.assertTrue(os.path.isfile(os.path.join(AdaPaSystemTest.result_dir, "pytorch_ns_tracing_0.json")))
        self.assertTrue(os.path.isfile(os.path.join(AdaPaSystemTest.result_dir, "pytorch_ns_summary_0.csv")))
        self.assertTrue(os.path.isfile(os.path.join(AdaPaSystemTest.result_dir, "pytorch_ns_op_stat_0.csv")))

    def test_report_trace(self):
        with patch.object(sys, 'argv', ["ada-pa",
                                        os.path.join(AdaPaSystemTest.log_dir, "pytorch_ns.log"),
                                        "--output={}".format(AdaPaSystemTest.result_dir),
                                        "--reporter=trace"]):
            self.assertEqual(ada_prof_cmd.main(), 0)
        result_file_path = os.path.join(AdaPaSystemTest.result_dir, "pytorch_ns_tracing_0.json")
        self.assertTrue(os.path.isfile(result_file_path))

        self.assert_show_in_ns(result_file_path)
        self.assert_records_num(result_file_path, 249)


if __name__ == '__main__':
    unittest.main()
