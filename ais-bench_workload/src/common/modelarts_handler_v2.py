import logging
import os
import time

from modelarts.estimatorV2 import JOB_STATE, Estimator
from modelarts.session import Session
from modelarts.train_params import InputData, OutputData, TrainingFiles
from obs import ObsClient

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_config_value(config, key):
    return None if config.get(key) == "" else config.get(key)


try:
    import moxing as mox
    moxing_import_flag = True
except Exception:
    moxing_import_flag = False


class modelarts_handler_v2():
    RESP_OK = 300

    def __init__(self):
        self.output_url = None
        self.job_log_prefix = None
        self.job_name = ""
        self.job_instance = None

    def sync_job_log(self, session_config):
        dstpath = os.path.join(os.getenv("BASE_PATH", "./"), "log")
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        for id in range(session_config.train_instance_count):
            logurl = self.job_log_prefix + '-' + str(id) + '.log'
            logname = os.path.basename(logurl)
            logpath = os.path.join(dstpath, logname)
            if self.session.obs.is_obs_path_exists(logurl):
                self.session.obs.download_file(logurl, logpath)

    def continue_waiting(self):
        print("waiting for task, phase %s, total time: %d(s)" % (self.job_info['status']['phase'],
                                                                 self.job_info['duration'] / 1000))

    def exit_by_failure(self):
        print("task failed, phase %s, please check log on obs, exit" % (self.job_info['status']['phase']))
        raise RuntimeError('job failed')

    def wait_for_job(self, session_config):
        count = 0
        while True:
            time.sleep(10)
            count = count + 1
            if count > 10:
                self.sync_job_log(session_config)
            job_info = self.job_instance.get_job_info()

            phase = job_info['status']['phase']
            if phase == "Completed":
                logger.info("task succeeded, total time %d(s)" % (job_info['status']['duration'] / 1000))
                break
            elif phase == 'Failed' or phase == 'Abnormal':
                self.exit_by_failure()
            else:
                self.continue_waiting()

    def create_obs_output_dirs(self, output_url):
        """
        output_url , like '/zgwtest/lcm_test/result/XXX'. bucket name: zgwtest subdir: lcm_test/result/XXX
        """
        # print("\n===================output_url: {} moxing_import_flag : {}".format(output_url, moxing_import_flag))
        if moxing_import_flag:
            dstpath = output_url.replace("s3:", "obs:", 1)
            logger.info("create obs outdir mox mkdir:{}".format(dstpath))
            mox.file.make_dirs(dstpath)
        else:
            bucket_name = output_url[1:].split('/')[0]
            sub_dir = output_url.replace(f"s3://{bucket_name}/", "", 1)
            logger.debug('create obs output{} subdir:{} bucket:{}'.format(output_url, sub_dir, bucket_name))
            resp = self.obsClient.putContent(bucket_name, sub_dir, content=None)

            if resp.status < self.RESP_OK:
                logger.debug('obs put content request ok')
            else:
                logger.warn('create obs folder failed. errorCode:{} msg:{}'.format(resp.errorCode, resp.errorMessage))
                raise RuntimeError('create obs folder failed')

    def create_obs_handler(self, access_config):
        if not moxing_import_flag:
            # Create OBS login handle
            self.obsClient = ObsClient(access_key_id=access_config.access_key,
                                       secret_access_key=access_config.secret_access_key, server=access_config.server)

    def create_session(self, access_config):
        # 如下配置针对计算中心等专有云 通用云不需要设置
        if access_config.get("iam_endpoint") != "" and access_config.get("iam_endpoint") is not None \
            and access_config.get("obs_endpoint") != "" and access_config.get("obs_endpoint") is not None \
            and access_config.get("modelarts_endpoint") != "" and access_config.get("modelarts_endpoint") is not None:
            Session.set_endpoint(iam_endpoint=access_config.iam_endpoint, obs_endpoint=access_config.obs_endpoint,
                                 modelarts_endpoint=access_config.modelarts_endpoint,
                                 region_name=access_config.region_name)
        # Create modelars handle
        self.session = Session(access_key=access_config.access_key,
                               secret_key=access_config.secret_access_key,
                               project_id=access_config.project_id,
                               region_name=access_config.region_name)

    def print_train_instance_types(self):
        algo_info = Estimator.get_train_instance_types(self.session)
        print("get valid train_instance_types:{}".format(algo_info))

    def stop_job(self):
        self.job_instance.control_job()

    def get_obs_url_content(self, obs_url):
        if moxing_import_flag:
            dsturl = obs_url.replace("s3:", "obs:", 1)
            with mox.file.File(dsturl, 'r') as f:
                file_str = f.read()
                return file_str
        else:
            bucket_name = obs_url[5:].split('/')[0]
            obs_sub_path = obs_url.replace(f"/{bucket_name}/", "", 1)
            resp = self.obsClient.getObject(bucket_name, obs_sub_path, loadStreamInMemory=True)

            if resp.status < self.RESP_OK:
                logger.debug('request ok')
                return resp.body.buffer.decode("utf-8")
            else:
                raise RuntimeError('obs get object ret:{} url:{} bucket:{} path:{}'.format(resp.status, obs_url,
                                                                                           bucket_name, obs_sub_path))

    def update_code_to_obs(self, session_config, localpath):
        if moxing_import_flag:
            dstpath = "obs:/" + session_config.code_dir
            logger.info("mox update loaclpath:{} dstpath:{}".format(localpath, dstpath))
            mox.file.copy_parallel(localpath, dstpath)
        else:
            bucket_name = session_config.code_dir.split('/')[1]
            sub_dir = "/".join(session_config.code_dir.strip("/").split('/')[1:])
            logger.info("update code localpath:{} codepath:{} bucket:{} subdir:{}".format(
                localpath, session_config.code_dir, bucket_name, sub_dir))
            resp = self.obsClient.putFile(bucket_name, sub_dir, localpath)
            if resp.status < self.RESP_OK:
                logger.info("update code to obs success. requestId:{}".format(resp.requestId))
            else:
                logger.error("update code to obs failed. errorCode:{} errorMessage:{}".format(resp.errorCode,
                                                                                              resp.errorMessage))
                raise RuntimeError('update code to obs failed')

    def create_modelarts_job(self, session_config, output_url):
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.job_name = session_config.job_name + "_ais-bench_" + timestr
        jobdesc = session_config.job_description_prefix + "_jobname_" + self.job_name + "_" +\
            str(session_config.train_instance_type) + "_" + str(session_config.train_instance_count)

        output_list = [OutputData(obs_path="obs:/" + session_config.out_base_url, name="train_url")]

        estimator = Estimator(session=self.session,
                              framework_type=session_config.framework_type,
                              framework_version=session_config.framework_version,
                              training_files=TrainingFiles(code_dir="obs:/" + session_config.code_dir,
                                                           boot_file="obs:/" + session_config.boot_file),
                              log_url="obs:/" + output_url,
                              parameters=session_config.parameters,
                              outputs=output_list,
                              pool_id=get_config_value(session_config, "pool_id"),
                              train_instance_type=get_config_value(session_config, "train_instance_type"),
                              train_instance_count=session_config.train_instance_count,
                              job_description=jobdesc,
                              user_command=None)

        logger.debug("new create inputs:{} job_name:{}".format(session_config.inputs, self.job_name))
        inut_list = [InputData(obs_path="obs:/" + session_config.inputs, name="data_url")]
        try:
            job_instance = estimator.fit(inputs=inut_list, wait=False, job_name=self.job_name)
        except Exception as e:
            logger.error("failed to create job on modelarts, msg %s" % (e))
            raise RuntimeError('creat job failed')

        logger.debug("inputs:{} job_name:{} ret instance:{}".format(inut_list, self.job_name, job_instance))
        job_info = job_instance.get_job_info()
        print("\njob_info: {}\n".format(job_info))

        if 'error_msg' in job_info.keys():
            logger.error("failed to run job on modelarts, msg %s" % (job_info['error_msg']))
            raise RuntimeError('creat job failed')

        self.job_log_prefix = output_url + self.job_name
        print("create job sucess. id:{}  name:{} create_time:{}".format(
              job_info["metadata"]["id"],  job_info["metadata"]["name"], job_info["metadata"]["create_time"]))
        return job_instance

    def run_job(self, session_config, localpath):
        logger.debug("session config:{}".format(session_config))

        self.print_train_instance_types()
        self.output_url = os.path.join(session_config.out_base_url + "lhbtest/")
        logger.debug("output_url:{}".format(self.output_url))
        self.create_obs_output_dirs(self.output_url)

        # update codes to obs
        self.update_code_to_obs(session_config, localpath)
        # create job
        job_instance = self.create_modelarts_job(session_config, self.output_url)
        self.job_instance = job_instance
        self.wait_for_job(session_config)
