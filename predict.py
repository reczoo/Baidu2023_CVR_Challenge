import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import model
import gc
import argparse
import os
from pathlib import Path
import pandas as pd


if __name__ == '__main__':
    ''' Usage: python predict.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    params["train_data"] = os.path.join(data_dir, 'train.h5')
    params["valid_data"] = os.path.join(data_dir, 'valid.h5')
    params["test_data"] = os.path.join(data_dir, 'test.h5')
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(model, params['model'])
    model = model_class(feature_map, **params)
    model.load_weights(model.checkpoint)

    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    y_pred = model.predict(test_gen)
    test_df = pd.read_csv("./2023-cvr-contest-data/data_v1/test.csv")
    test_df["predict"] = y_pred
    test_df["predict"] = test_df["predict"].map(lambda x: "{:.16f}".format(x))
    test_df[["log_id", "predict"]].to_csv("test-1.txt", header=False, index=False)
