from easydict import EasyDict as ed

access_config = ed({
    # 登录需要的ak sk信息
    'access_key': '',
    'secret_access_key': '',
    # 连接OBS的服务地址。可包含协议类型、域名、端口号。（出于安全性考虑，建议使用https协议）
    'server': '',
    # project_id/region_name:
    # 项目ID/区域ID，获取方式参考链接
    # https://support.huaweicloud.com/api-iam/iam_17_0002.html
    'region_name': '',
    'project_id': '',

    # # 如下配置针对计算中心等专有云 通用云不需要设置 设置为空
    'iam_endpoint': '',
    'obs_endpoint': '',
    'modelarts_endpoint' : '',
})

session_config = ed({
    'hyperparameters': [
        {'label': 'config_path', 'value': 'resnet50_imagenet2012_Acc_config.yaml'},
        {'label': 'enable_modelarts', 'value': 'True'},
        {'label': 'run_distribute', 'value': 'True'},
        {'label': 'epoch_size', 'value': '5'},
        {'label': 'device_num', 'value': '8'},
        {'label': 'run_eval', 'value': 'True'},
    ],
    'inputs': '/0923/zhou/datasets/small/',
    'code_dir': '/0923/00lcm/lcmtest/resnet/',
    'boot_file': '/0923/00lcm/lcmtest/resnet/train.py',

    # 如下为运行相关参数
    # job名称 如果存在就增加版本创建
    'job_name': "aisbench-debug",

    # 使用容器类型与镜像版本
    'framework_type': 'Ascend-Powered-Engine',
    'framework_version': 'MindSpore-1.3-cann_5.0.2-python3.7-euleros2.8-aarch64',

    # 训练类型 如下为8卡 其他类型待补充 待确定与flavor中差异
    'train_instance_type': 'modelarts.kat1.8xlarge',
    # 训练结点数
    'train_instance_count': 2,
    # 输出信息基准路径 整体路径为 train_url = out_base_url/version_name
    "out_base_url": "/0923/00lcm/result_dump/res",
    # job 描述前缀
    "job_description_prefix": 'lcm-debug desc',
})