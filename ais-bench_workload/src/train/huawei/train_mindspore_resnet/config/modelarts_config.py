from easydict import EasyDict as ed

# 该部分为认证信息，请向相关运维同事咨询并填写
access_config = ed({
    # 登录需要的ak sk信息
    'access_key': '',
    'secret_access_key': '',
    # 连接OBS的服务地址。可包含协议类型、域名、端口号。（出于安全性考虑，建议使用https协议）
    # 如果是计算中心，需要联系运维同事获取
    'server': '',
    # project_id/region_name:
    # 项目ID/区域ID，获取方式参考链接
    # https://support.huaweicloud.com/api-iam/iam_17_0002.html
    # 如果是计算中心,请咨询相关维护同事
    'region_name': '',
    'project_id': '',

    # 如下配置针对计算中心等专有云 通用云不需要设置 设置为空 请咨询相关维护同事
    # 设置该信息后 需要设置相关的域名解析地址
    'iam_endpoint': '',
    'obs_endpoint': '',
    'modelarts_endpoint' : '',
})

session_config = ed({
    # 运行模型的传入超参
    'hyperparameters': [
        # 模型配置文件，默认boost模式，不需要更改
        {'label': 'config_path', 'value': 'resnet50_imagenet2012_Boost_config.yaml'},
        # 是否使能modelarts 必须设置为True，不需要修改
        {'label': 'enable_modelarts', 'value': 'True'},
        # 是否开启分布式，如果1卡以上的话都是True 一般不需要修改
        {'label': 'run_distribute', 'value': 'True'},
        # epoch次数 必须关注 当前默认设置为90
        {'label': 'epoch_size', 'value': '90'},
        # device数量 云上场景一般不需要修改
        {'label': 'device_num', 'value': '8'},
        # 是否保存ckpt文件 默认为True 保存ckpt
        {'label': 'save_checkpoint', 'value': 'False'},
        # 保存ckpt的epoch数 必须修改并注意 该值必须要跟epoch数一致 这样提高性能
        {'label': 'save_checkpoint_epochs', 'value': '90'},
    ],
    # 输入数据集obs目录,请按样例格式填写
    'inputs': '/zgwtest/lcm_test/dataset/imagenet_small/',
    # obs代码路径 程序会自动拷贝到该路径
    'code_dir': '/zgwtest/lcm_test/resnet/',
    # 启动文件 必须要在code_dir路径下，请按样例格式填写
    'boot_file': '/zgwtest/lcm_test/resnet/train.py',

    # 如下为运行相关参数
    # job名称  如果云环境Modelarts服务训练作业job队列中没有，则会新建一个job；若和已有job同名，则会在该job中，新建测试实例.
    'job_name': "aisbench-debug",

    # 使用容器类型与镜像版本
    'framework_type': 'Ascend-Powered-Engine',
    'framework_version': 'MindSpore-1.3-cann_5.0.2-python3.7-euleros2.8-aarch64',

    # 资源参数类型主要包括如下2个值 train_instance_type和pool_id
    # 不设置pool_id 默认是公共池 设置了就是专属资源池
    # 只设置pool_id 不设置train_instance_type 默认为专属资源池的默认类型
    # train_instance_type 在程序打印中有提示的 一般为如下四个值 分别对应 1卡 2卡 4卡 8卡
    # ['modelarts.kat1.xlarge', 'modelarts.kat1.2xlarge', 'modelarts.kat1.4xlarge', 'modelarts.kat1.8xlarge']
    # https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0191.html 该链接指示获取方法

    # 专属资源池id 不是则为None
    'pool_id': None,
    # 训练类型 如下为8卡 如果是专属资源池id设置，那么该类型需要设置为None
    'train_instance_type': 'modelarts.kat1.8xlarge',
    # 训练结点数
    'train_instance_count': 1,

    # 云存储路径 默认为空
    # 'nas_type' : None,
    # 'nas_share_addr' : None,
    # 'nas_mount_path' : None,

    # 输出信息基准路径 整体路径为 train_url = out_base_url/version_name
    "out_base_url": "/zgwtest/lcm_test/result/",
    # job 描述前缀
    "job_description_prefix": 'lcm-debug desc',
})

session_config_v2 = ed({
    # 运行模型的传入超参
    'parameters': [
        # 模型配置文件，默认boost模式，不需要更改
        {'name': 'config_path', 'value': 'resnet50_imagenet2012_Boost_config.yaml'},
        # 是否使能modelarts 必须设置为True，不需要修改
        {'name': 'enable_modelarts', 'value': 'True'},
        # 是否开启分布式，如果1卡以上的话都是True 一般不需要修改
        {'name': 'run_distribute', 'value': 'True'},
        # epoch次数 必须关注 当前默认设置为90
        {'name': 'epoch_size', 'value': '90'},
        # device数量 云上场景一般不需要修改
        {'name': 'device_num', 'value': '8'},
        # 是否保存ckpt文件 默认为True 保存ckpt
        {'name': 'save_checkpoint', 'value': 'False'},
        # 保存ckpt的epoch数 必须修改并注意 该值必须要跟epoch数一致 这样提高性能
        {'name': 'save_checkpoint_epochs', 'value': '90'},
    ],
    # 输入数据集obs目录,请按样例格式填写
    'inputs': '/zgwtest/lcm_test/dataset/imagenet_small/',

    # obs代码路径 程序会自动拷贝到该路径. 和boot_files一起用于复合参数 training_files
    'code_dir': '/zgwtest/lcm_test/resnet/',
    # 启动文件 必须要在code_dir路径下，请按样例格式填写
    'boot_file': '/zgwtest/lcm_test/resnet/train.py',

    # 如下为运行相关参数
    # job名称  如果云环境Modelarts服务训练作业job队列中没有，则会新建一个job；若和已有job同名，则会在该job中，新建测试实例.
    'job_name': "aisbench-debug",

    # 使用容器类型与镜像版本
    'framework_type': 'Ascend-Powered-Engine',
    'framework_version': 'mindspore_1.3.0-cann_5.0.2-py_3.7-euler_2.8.3-aarch64',

    # 资源参数类型主要包括如下2个值 train_instance_type和pool_id
    # 不设置pool_id 默认是公共池 设置了就是专属资源池
    # 只设置pool_id 不设置train_instance_type 默认为专属资源池的默认类型
    # 专属资源池id. 该变量设置为"None"，表示使用公共资源池训练。专属资源池id获取，可以从云环境modelarts服务训练管理界面手工创
    # 建新的训练作业，查看“专属资源池"是否资源。有，取其id作为pool_id，注意有的云环境需要剔除间隔符，比如"-"。没有专属资源池时，请自行申请购买资源
    'pool_id': None,
    # train_instance_type 在程序打印中有提示的,请注意紧随“get valid train_instance_types：”之后的打印输出. 由modelarts.estimatorV2 类Estimator的接口get_train_instance_types（）查询而来。
    # 请参见https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0431.html 该链接指示获取方法。注意不同云环境查询的结果不同
    # 训练类型 如下为8卡。 如果专属资源池id设置了，那么该类型一般需要设置为None，但有的云环境则不需要，设置了可能会报错，请根据具体环境进行设置
    'train_instance_type': 'modelarts.kat1.8xlarge',
    # 训练结点数
    'train_instance_count': 1,

    # 云存储路径 默认为空
    # 'nas_type': None,
    # 'nas_share_addr': None,
    # 'nas_mount_path': None,

    # 输出信息基准路径 整体路径为 train_url = out_base_url/version_name
    "out_base_url": "/zgwtest/lcm_test/result/",
    # job 描述前缀
    "job_description_prefix": 'lcm-debug desc',
})
