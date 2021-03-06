import os
from statistics import mean
import logging
import argparse
from config.modelarts_config import access_config, session_config
from modelarts_handler import modelarts_handler,logger
import ais_utils

def report_result(handler):
    ranksize_file_url = os.path.join(handler.output_url, 'ranksize.json')
    ranksize = int(handler.get_obs_url_content(ranksize_file_url))
    print("url:{} read ranksize:{}".format(ranksize_file_url, ranksize))

    total_throughput = 0.0
    for rankid in range(0, ranksize):
        throughput_url = os.path.join(handler.output_url, 'throughput_' + str(rankid) + '.json')
        single_throughput_rate = float(handler.get_obs_url_content(throughput_url))
        print("rankid:{} url:{} read throughput:{}".format(rankid, throughput_url, single_throughput_rate))
        total_throughput = total_throughput + single_throughput_rate
    print("report result total_throughput : {}".format(total_throughput))

    accuracy_file_url = os.path.join(handler.output_url, 'accuracy_0.json')
    accuracy = float(handler.get_obs_url_content(accuracy_file_url))
    print("url:{} read accuracy:{}".format(accuracy_file_url, accuracy))

    print("report result throughput:{} accuracy:{}".format(total_throughput, accuracy))
    ais_utils.set_result("training", "throughput_ratio", total_throughput)
    ais_utils.set_result("training", "accuracy", accuracy)

# 单设备运行模式
def report_result_singlesever_mode(handler, server_count):
    # 单设备运行模式下默认都是8卡
    cards_per_server = 8
    print("server_count:{} cards_per_server:{}".format(server_count, cards_per_server))

    throughput_list = []
    accuracy_list = []
    for server_id in range(server_count):
        single_server_throughput = 0.0
        for rankid in range(cards_per_server):
            throughput_url = os.path.join(handler.output_url, str(server_id), 'throughput_' + str(rankid) + '.json')
            single_card_throughput = float(handler.get_obs_url_content(throughput_url))
            print("rankid:{} url:{} read throughput:{}".format(rankid, throughput_url, single_card_throughput))
            single_server_throughput = single_server_throughput + single_card_throughput
        print("serverid:{} count:{} service_throughput:{}".format(server_id, server_count, single_server_throughput))
        throughput_list.append(single_server_throughput)

        accuracy_file_url = os.path.join(handler.output_url, 'accuracy_{}.json'.format(server_id))
        single_server_accuracy = float(handler.get_obs_url_content(accuracy_file_url))
        print("serverid:{} url:{} read accuracy:{}".format(server_id, accuracy_file_url, single_server_accuracy))
        accuracy_list.append(single_server_accuracy)

    print("report >> throughput_list:{} average:{}".format(throughput_list, mean(throughput_list)))
    print("report >> accuracy_list:{} average:{}".format(accuracy_list, mean(accuracy_list)))

    ais_utils.set_result("training", "throughput_ratio", mean(throughput_list))
    ais_utils.set_result("training", "accuracy", mean(accuracy_list))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_code_path", required=True, help="the local path of run code")
    parser.add_argument("--single_server_mode", action="store_true", help="the local path of run code")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    logger.setLevel(logging.DEBUG)

    handler = modelarts_handler()
    handler.create_obs_handler(access_config)
    handler.create_session(access_config)
    handler.run_job(session_config, args.local_code_path)

    # handler.output_url = "s3://0923/00lcm/result_dump/res/V212/"
    if args.single_server_mode == True:
        report_result_singlesever_mode(handler, session_config.train_instance_count)
    else:
        report_result(handler)
    