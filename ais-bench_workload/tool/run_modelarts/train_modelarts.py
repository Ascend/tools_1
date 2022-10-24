import argparse
import logging
import os
import sys
from statistics import mean

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from modelarts_config import access_config
from modelarts_config import session_config as session_config_v1

from modelarts_handler import logger, modelarts_handler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_code_path", default="", help="the local path of run code")
    parser.add_argument("--action", default="run", choices=["run", "stop"], help="action (run or stop)")
    parser.add_argument("--modelarts_version", default="V1", choices=["V1", "V2"], help="modelarts version (V1 or V2)")
    parser.add_argument("--job_id", default="None",  help="job id used to stop given job")
    parser.add_argument("--timeout", type=int, default=0,  help="timeout to stop job")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    logger.setLevel(logging.DEBUG)
    session_config = session_config_v1 if args.modelarts_version == 'V1' else None

    handler = modelarts_handler() if args.modelarts_version == 'V1' else None
    handler.create_session(access_config)

    if args.action == "stop":
        if args.modelarts_version == 'V1':
            handler.stop_new_versions(session_config)
        else:
            handler.stop_job(args.job_id)
        sys.exit()

    handler.create_obs_handler(access_config)

    # default run mode
    handler.run_job(session_config, args.local_code_path, args.timeout)