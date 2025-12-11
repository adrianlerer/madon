import importlib
import sys
import os
from ast import literal_eval

from networkx.algorithms.traversal import dfs_edges

# from madon.src.task_3.MLP.significance import Significance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-holistic-formalism-feature-based')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../task_3/MLP')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# print(os.listdir())
# print(sys.path)
# from data_manager import DataManager
from utils.utilities import *
from process_features import ProcessFeatures
input('yu see')
from configuration import Configuration
from train import Trainer

from processing.create_dataset import CreateDataset
from processing.data_statistics import DataStats


def __main__():
    # set and get the parameters to shape the frame of the project
    data_path = '../../data'
    ds_path = os.path.join(data_path, 'processed_datasets')
    stats_path = os.path.join(data_path, 'statistics')
    parameters = get_parameters()
    train_val = parameters['train']

    # initialize the configuration at backend (choice of device, fixing the seed)
    configuration = Configuration(parameters)
    # parameters.update(configuration.parameters)
    dataset_obj = CreateDataset(ds_path, parameters['is_meta'], parameters['split'])
    statistics_obj = DataStats(dataset_obj, parameters['is_meta'], stats_path)
    statistics_obj.process()

    # push_data_to_hub(data_manager)

    # process the features to create dataset for feature engineering
    proc_features = ProcessFeatures(dataset_obj, parameters=parameters)
    # input('..........')
    train_obj = Trainer(parameters, config_obj=configuration, feat_proc_obj=proc_features)
    train_obj.evaluate_best(split='test')

    # if you put inference=True, and eval_all True, it will run all combinations

    train_obj.evaluate_best(split='test', inference=parameters['inference_choice'])
    if parameters['eval_all']:
        train_obj.evaluate_all_scenarios(split='test', inference=parameters['inference_choice'])



if __name__ == '__main__':
    __main__()