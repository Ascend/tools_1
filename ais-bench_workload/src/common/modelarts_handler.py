import logging
import os
import time

from modelarts.estimator import JOB_STATE, Estimator
from modelarts.estimatorV2 import Estimator as Estimatorv2
from modelarts.session import Session
from modelarts.train_params import InputData, OutputData, TrainingFiles
from obs import ObsClient

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_config_value(config, key):
    return None if config.get(key) == "" else config.get(key)


def continue_waiting(job_info):
    print("waiting for task, status %s, total time: %d(s)" % (JOB_STATE[job_info['status']],
                                                              job_info['duration'] / 1000))


def exit_by_failure(job_info):
    print("task failed, status %s, please check log on obs, exit" % (JOB_STATE[job_info['status']]))
    raise RuntimeError('failed')


func_table = {
    0: continue_waiting,
    1: continue_waiting,
    2: continue_waiting,
    3: exit_by_failure,
    4: continue_waiting,
    5: exit_by_failure,
    6: exit_by_failure,
    7: continue_waiting,
    8: continue_waiting,
    9: exit_by_failure,
    11: exit_by_failure,
    12: exit_by_failure,
    13: exit_by_failure,
    14: exit_by_failure,
    15: continue_waiting,
    16: exit_by_failure,
    17: exit_by_failure,
    18: continue_waiting,
    19: continue_waiting,
    20: continue_waiting,
    21: exit_by_failure,
    22: exit_by_failure
}


# for debugging, stop after timeout
def wait_for_job_timeout(job_instance):
    count = 0
    while True:
        time.sleep(10)
        job_info = job_instance.get_job_info()
        if job_info['status'] == 10:
            print("task succeeded, total time %d(s)" % (job_info['duration'] / 1000))
            break
        func_table[job_info['status']](job_info)
        count = count + 1
        print("modelarts run time count:{}".format(count))
        if count == 6:
            print("modelarts run match:{} 10 so exit >>>>>>>".format(count))
            status = job_instance.stop_job_version()
            #status = job_instance.delete_job()
            raise RuntimeError('failed')
            break


try:
    import moxing as mox
    moxing_import_flag = True
except Exception:
    moxing_import_flag = False


class modelarts_handler():
    RESP_OK = 300
    FIX_VERSION = 1

    def __init__(self):
        self.output_url = None
        self.job_log_prefix = None
        self.job_name = ""
        self.job_instance = None
        self.session_config = None
        self.modelarts_version = 'V1'
        self.bucket_name = None

    def sync_job_log(self, session_config):
        dstpath = os.path.join(os.getenv("BASE_PATH", "./"), "log")
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        for id in range(session_config.train_instance_count):
            logurl = self.job_log_prefix + '-' + str(id) + '.log'
            logname = os.path.basename(logurl)
            logpath = os.path.join(dstpath, logname)
            #print("\n===============logurl: {} logname: {} logpath: {}".format(logurl, logname, logpath))
            if self.session.obs.is_obs_path_exists(logurl):
                self.session.obs.download_file(logurl, logpath)

    def wait_for_job(self):
        count = 0
        while True:
            time.sleep(10)
            count = count + 1
            if count > 10:
                # count = 10
                self.sync_job_log(self.session_config)
            job_info = self.job_instance.get_job_info()
            if self.modelarts_version == 'V1':
                if job_info['status'] == 10:
                    print("task succeeded, total time %d(s)" % (job_info['duration'] / 1000))
                    break
                func_table[job_info['status']](job_info)
            else:
                # V2
                phase = job_info['status']['phase']
                if phase == "Completed":
                    logger.info("task succeeded, total time %d(s)" % (job_info['status']['duration'] / 1000))
                    break
                elif phase in ['Failed', 'Abnormal', 'Terminated']:
                    print("task failed, phase %s, please check log on obs, exit" % (job_info['status']['phase']))
                    raise RuntimeError('job failed')
                else:
                    print("waiting for task, phase %s, total time: %d(s)" % (job_info['status']['phase'], 10 * count))

    def create_obs_output_dirs(self, output_url):
        if moxing_import_flag:
            dstpath = output_url.replace("s3:", "obs:", 1)
            logger.info("create obs outdir mox mkdir:{}".format(dstpath))
            mox.file.make_dirs(dstpath)
        else:
            sub_dir = output_url.replace(f"s3://{self.bucket_name}/", "", 1)
            logger.debug('create obs output{} subdir:{} bucket:{}'.format(output_url, sub_dir, self.bucket_name))
            resp = self.obsClient.putContent(self.bucket_name, sub_dir, content=None)
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
        algo_info = Estimator.get_train_instance_types(modelarts_session=self.session)
        print("get valid train_instance_types:{}".format(algo_info))

    def stop_new_versions_for(self, session_config):
        base_job_list_info = Estimator.get_job_list(modelarts_session=self.session, per_page=10, page=1, order="asc",
                                                    search_content=session_config.job_name)
        if base_job_list_info is None or base_job_list_info.get("job_total_count", 0) == 0:
            print("find no match version return")
        else:
            pre_version_id = base_job_list_info["jobs"][0].get("version_id")
            job_id = base_job_list_info["jobs"][0].get("job_id")
            job_status = base_job_list_info["jobs"][0].get("status")
            estimator = Estimator(modelarts_session=self.session, job_id=job_id, version_id=pre_version_id)
            if JOB_STATE[job_status] == "JOBSTAT_INIT" \
                or JOB_STATE[job_status] == "JOBSTAT_IMAGE_CREATING" \
                or JOB_STATE[job_status] == "JOBSTAT_SUBMIT_TRYING" \
                or JOB_STATE[job_status] == "JOBSTAT_DEPLOYING" \
                or JOB_STATE[job_status] == "JOBSTAT_WAITING" \
                or JOB_STATE[job_status] == "JOBSTAT_RUNNING":
                status = estimator.stop_job_version()
                print("jobname:{} jobid:{} preversionid:{} jobstatus:{} stop status:{}".format(
                    session_config.job_name, job_id, pre_version_id, JOB_STATE[job_status], status))
            else:
                print("jobname:{} jobid:{} preversionid:{} jobstatus:{} no need stop".format(
                    session_config.job_name, job_id, pre_version_id, JOB_STATE[job_status]))

    def stop_job(self):
        if self.modelarts_version == 'V1':
            self.stop_new_versions_for(self.session_config)
        else:
            self.job_instance.control_job()

    def get_job_name_next_new_version(self):
        base_job_list_info = Estimator.get_job_list(modelarts_session=self.session, per_page=10, page=1, order="asc",
                                                    search_content=self.session_config.job_name)
        if base_job_list_info is None or base_job_list_info.get("job_total_count", 0) == 0:
            return self.FIX_VERSION
        else:
            pre_version_id = base_job_list_info["jobs"][0].get("version_id")
            job_id = base_job_list_info["jobs"][0].get("job_id")
            estimator = Estimator(modelarts_session=self.session, job_id=job_id, version_id=pre_version_id)
            job_info = estimator.get_job_info()
            pre_version_id = job_info.get("version_name", "V0")[1:]
            return int(pre_version_id)+1

    def get_obs_url_content(self, obs_url):
        if moxing_import_flag:
            dsturl = obs_url.replace("s3:", "obs:", 1)
            with mox.file.File(dsturl, 'r') as f:
                file_str = f.read()
                return file_str
        else:
            if self.modelarts_version == 'V1':
                obs_sub_path = obs_url.replace(f"s3://{self.bucket_name}/", "", 1)
            else:
                obs_sub_path = obs_url.replace(f"/{self.bucket_name}/", "", 1)

            resp = self.obsClient.getObject(self.bucket_name, obs_sub_path, loadStreamInMemory=True)
            if resp.status < self.RESP_OK:
                logger.debug('request ok')
                return resp.body.buffer.decode("utf-8")
            else:
                raise RuntimeError('obs get object ret:{} url:{} bucket:{} \
                                   path:{}'.format(resp.status, obs_url, self.bucket_name, obs_sub_path))

    def update_code_to_obs(self, localpath):
        if moxing_import_flag:
            dstpath = "obs:/" + self.session_config.code_dir
            logger.info("mox update loaclpath:{} dstpath:{}".format(localpath, dstpath))
            mox.file.copy_parallel(localpath, dstpath)
        else:
            sub_dir = "/".join(self.session_config.code_dir.strip("/").split('/')[1:])
            logger.info("update code localpath:{} codepath:{} bucket:{} subdir:{}".format(
                localpath, self.session_config.code_dir, self.bucket_name, sub_dir))
            print("\n==============self.bucket_name:{} sub_dir: {} localpath:{}".format(self.bucket_name, sub_dir, localpath))
            resp = self.obsClient.putFile(self.bucket_name, sub_dir, localpath)
            # if resp.status < self.RESP_OK:
            #     logger.info("update code to obs success. requestId:{}".format(resp.requestId))
            # else:
            #     logger.error("update code to obs failed. errorCode:{} errorMessage:{}".format(resp.errorCode,
            #                                                                                   resp.errorMessage))
            #     raise RuntimeError('update code to obs failed')

    def create_modelarts_job_v1(self, output_url):
        jobdesc = self.session_config.job_description_prefix + "_jobname_" + self.session_config.job_name + "_" +\
            str(self.session_config.train_instance_type) + "_" + str(self.session_config.train_instance_count)
        estimator = Estimator(modelarts_session=self.session,
                              framework_type=self.session_config.framework_type,
                              framework_version=self.session_config.framework_version,
                              code_dir=self.session_config.code_dir,
                              boot_file=self.session_config.boot_file,
                              log_url=output_url[4:],
                              hyperparameters=self.session_config.hyperparameters,
                              output_path=output_url[4:],
                              pool_id=get_config_value(self.session_config, "pool_id"),
                              train_instance_type=get_config_value(self.session_config, "train_instance_type"),
                              train_instance_count=self.session_config.train_instance_count,
                              nas_type=get_config_value(self.session_config, "nas_type"),
                              nas_share_addr=get_config_value(self.session_config, "nas_share_addr"),
                              nas_mount_path=get_config_value(self.session_config, "nas_mount_path"),
                              job_description=jobdesc,
                              user_command=None)

        base_job_list_info = Estimator.get_job_list(modelarts_session=self.session, per_page=10, page=1, order="asc",
                                                    search_content=self.session_config.job_name)
        if base_job_list_info is None or base_job_list_info.get("job_total_count", 0) == 0:
            logger.debug("new create inputs:{} job_name:{}".format(self.session_config.inputs,
                                                                   self.session_config.job_name))
            job_instance = estimator.fit(inputs=self.session_config.inputs, wait=False,
                                         job_name=self.session_config.job_name)
        else:
            job_id = base_job_list_info["jobs"][0].get("job_id")
            pre_version_id = base_job_list_info["jobs"][0].get("version_id")
            logger.debug("new versions job_id:{} pre_version_id:{}".format(job_id, pre_version_id))
            job_instance = estimator.create_job_version(job_id=job_id, pre_version_id=pre_version_id,
                                                        inputs=self.session_config.inputs, wait=False, job_desc=jobdesc)

        print("inputs:{} job_name:{} ret instance:{}".format(self.session_config.inputs, self.session_config.job_name,
                                                             self.job_instance))
        job_info = job_instance.get_job_info()
        if not job_info['is_success']:
            logger.error("failed to run job on modelarts, msg %s" % (job_info['error_msg']))
            raise RuntimeError('failed')

        self.job_log_prefix = "obs:/" + output_url[4:] + job_info["resource_id"] + "-job-" + \
            self.session_config.job_name

        print("create sucess job_id:{} resource_id:{} version_name:{} create_time:{}".format(
            job_info["job_id"], job_info["resource_id"], job_info["version_name"], job_info["create_time"]))
        return job_instance

    def create_modelarts_job_v2(self, output_url):
        jobdesc = self.session_config.job_description_prefix + "_jobname_" + self.job_name + "_" +\
            str(self.session_config.train_instance_type) + "_" + str(self.session_config.train_instance_count)

        output_list = [OutputData(obs_path="obs:/" + self.session_config.out_base_url + self.job_name + "/",
                                  name="train_url")]

        estimator = Estimatorv2(session=self.session,
                                framework_type=self.session_config.framework_type,
                                framework_version=self.session_config.framework_version,
                                training_files=TrainingFiles(code_dir="obs:/" + self.session_config.code_dir,
                                                             boot_file="obs:/" + self.session_config.boot_file),
                                log_url="obs:/" + output_url,
                                parameters=self.session_config.parameters,
                                outputs=output_list,
                                pool_id=get_config_value(self.session_config, "pool_id"),
                                train_instance_type=get_config_value(self.session_config, "train_instance_type"),
                                train_instance_count=self.session_config.train_instance_count,
                                job_description=jobdesc,
                                user_command=None)

        logger.debug("new create inputs:{} job_name:{}".format(self.session_config.inputs, self.job_name))
        inut_list = [InputData(obs_path="obs:/" + self.session_config.inputs, name="data_url")]
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

        self.job_log_prefix = 'obs:/' + output_url + "modelarts-job-" + job_info['metadata']['id'] + '-worker'
        print("create job sucess. id:{}  name:{} create_time:{} job_log_prefix:{}".format(
              job_info["metadata"]["id"],  job_info["metadata"]["name"], job_info["metadata"]["create_time"],
              self.job_log_prefix))

        return job_instance

    def create_modelarts_job(self, output_url):
        if self.modelarts_version == 'V1':
            job_instance = self.create_modelarts_job_v1(output_url)
        else:
            job_instance = self.create_modelarts_job_v2(output_url)
        return job_instance

    def run_job(self, localpath):
        logger.debug("session config:{}".format(self.session_config))
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.job_name = self.session_config.job_name + timestr
        # self.print_train_instance_types()
        # modelarts path end with '/'，or report error ModelArts.2791
        if self.modelarts_version == 'V1':
            # get job_name's next version number
            next_version_id = self.get_job_name_next_new_version()
            # generate output path
            self.output_url = os.path.join("s3:/{}".format(self.session_config.out_base_url),
                                           "V{}".format(next_version_id), "")
        else:
            self.output_url = os.path.join(self.session_config.out_base_url, self.job_name, "")
        self.bucket_name = self.session_config.out_base_url.split('/')[1]
        logger.debug("output_url:{}".format(self.output_url))
        self.create_obs_output_dirs(self.output_url)

        # update code to obs
        self.update_code_to_obs(localpath)

        self.job_instance = self.create_modelarts_job(self.output_url)
        self.wait_for_job()
