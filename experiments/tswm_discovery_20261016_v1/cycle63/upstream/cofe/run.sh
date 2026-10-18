export CUDA_VISIBLE_DEVICES=0
MODEL=COFE
DATASET=etth1
CTX_LEN=96
PRED_LEN=720

DATA_DIR=./datasets
LOG_DIR=./log_dir


# Long-term forecasting:
# ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']
# NOTE: when using long-term forecasting datasets, please explicit assign context_length and prediction_length, e.g., :
# --data.data_manager.init_args.context_length 96 \
# --data.data_manager.init_args.prediction_length 720 \

# run pipeline with train and test
# replace ${MODEL} with tarfet model name, e.g, patchtst
# replace ${DATASET} with dataset name

# if not specify dataset_path, the default path is ./datasets

python run.py --config config/ltsf/${DATASET}/${MODEL}.yaml --seed_everything 0  \
    --data.data_manager.init_args.path ${DATA_DIR} \
    --trainer.default_root_dir ${LOG_DIR} \
    --data.data_manager.init_args.dataset ${DATASET} \
    --data.data_manager.init_args.split_val true \
    --trainer.max_epochs 50 \
    --data.data_manager.init_args.context_length ${CTX_LEN} \
    --data.data_manager.init_args.prediction_length ${PRED_LEN} 