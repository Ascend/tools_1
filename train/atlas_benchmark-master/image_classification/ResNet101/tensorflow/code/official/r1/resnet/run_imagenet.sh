export RANK_ID=$1
export RANK_SIZE=$2

export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
export RANK_TABLE_FILE=rank_table.json

export JOB_ID=10087
export FUSION_TENSOR_SIZE=1000000000
#sleep 5
python3 $3/imagenet_main.py \
--resnet_size=101 \
--batch_size=128 \
--num_gpus=1 \
--cosine_lr=True \
--dtype=fp16 \
--label_smoothing=0.1 \
--loss_scale=512 \
--train_epochs=3 \
--epochs_between_evals=1 \
--hooks=ExamplesPerSecondHook,loggingtensorhook,loggingmetrichook \
--data_dir=/home/imagenet_TF \
--model_dir=./model_dir

mkdir graph
mv *.txt graph
mv *.pbtxt graph
