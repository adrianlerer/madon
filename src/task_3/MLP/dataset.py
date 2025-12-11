from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from configuration import Configuration
from process_features import ProcessFeatures
import torch
import os
import pickle
import numpy as np

class FeatSet(Dataset):
    def __init__(self, process_obj: ProcessFeatures, split: str, config_obj: Configuration, normalize: bool=False, inference: bool=False, seed_info: int=42):
        """
        initializer of the FeatSet class
        :param process_obj: Feature Processing object
        :param split: string to specify which split to use
        :param config_obj: configuration object which allows us to reach the configurational setup
        :param normalize: boolean to decide whether to normalize the dataset or not
        """
        self.process_obj = process_obj
        self.config_obj = config_obj
        self.inference = inference
        self.seed_info = seed_info
        self.split = split

        self.data, self.labels = self.get_data(normalize=normalize)

    def set_scaler(self, dataset: np.ndarray) -> StandardScaler:
        """
        Method to set a scaler for the dataset
        :param dataset: a dataset which will be scaled (if train scaler is set, otherwise it is fit)
        :return: Standard Scaler to be used to fit the other datasets (val and test) to the train's distribution
        """
        scaler_path = os.path.join(self.process_obj.dataset_obj.result_path, 'scaler.pickle')
        if not os.path.exists(scaler_path):
            if not self.split == 'train':
                raise RuntimeError('scaler must be set in train mode')
            scaler = StandardScaler()
            scaler.fit(dataset)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def normalize(self, dataset: np.ndarray) -> np.ndarray:
        """
        Method to normalize the dataset
        :param dataset: numpy array for the specific dataset
        :return: normalized dataset
        """
        scaler = self.set_scaler(dataset)
        return scaler.transform(dataset)

    def get_data(self, normalize: bool=False) -> tuple:
        """
        Method to get the data
        :param normalize: boolean variable to decide whether to normalize the dataset or not
        :return: input data and ground truth labels
        """
        dataset, labels = self.process_obj.get_dataset(ds_type=self.split, inference=self.inference, seed_info=self.seed_info)
        data = self.normalize(dataset) if normalize else dataset
        return torch.FloatTensor(data), torch.LongTensor(labels)

    def __len__(self) -> int:
        """
        Method to return the length of the dataset
        :return: integer the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Method to get the data and ground truth labels
        :param idx: specifies the index of the data
        :return: dictionary with the data and ground truth label
        """
        return {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
