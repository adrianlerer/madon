import pickle

import torch
import os
import matplotlib.pyplot as plt
from lime import lime_tabular, submodular_pick
import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from train import Trainer
from configuration import Configuration
from process_features import ProcessFeatures
import itertools

class Significance:
    def __init__(self, trainer_obj: Trainer, parameters: dict, config_obj: Configuration, features_obj: ProcessFeatures):
        """
        initializer for significance class
        :param trainer_obj: trainer object to train the model
        :param parameters: dictionary to hold the parameters
        :param config_obj: configuration object to reach the experiment setup
        :param features_obj: feature processing object
        """
        self.trainer_obj = trainer_obj
        self.parameters = parameters
        self.config_obj = config_obj
        self.features_obj = features_obj
        self.scaler = self.set_scaler()
        self.report_names = self.beautify()
        self.model = None
        self.set_model()
        self.lime_explainer = None
        self.shap_explainer = None
        self.explanations_path = None
        self.set_explainer()

    @staticmethod
    def check_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def set_scaler(self) -> StandardScaler:
        """
        Method to set the scaler
        :return: either none or scaler
        """
        if not self.parameters['normalize']:
            print('INFO: No scaler was set, since it is not required')
            return None
        scaler_path = os.path.join(self.features_obj.dataset_obj.result_path, "scaler.pickle")
        # print(scaler_path)
        if not os.path.exists(scaler_path):
            raise NotADirectoryError('You need to have a scaler file to perform this operation')

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def beautify(self):
        # return {
        #     'doc_len_tokens': 'Document length',
        #     'num_args': "Number of arguments",
        #     'avg_arg_len': 'Average argument length per document',
        #     'frac_arg_CL': 'Fraction of argument CL',
        #     'frac_arg_D': 'Fraction of argument D',
        #     'frac_arg_PL': 'Fraction of argument PL',
        #     'frac_arg_HI': 'Fraction of argument HI',
        #     'frac_arg_LIN': 'Fraction of argument LIN',
        #     'frac_arg_PC': 'Fraction of argument PC',
        #     'frac_arg_TI': 'Fraction of argument TI',
        #     'frac_arg_SI': 'Fraction of argument SI',
        #     'O - OVERALL - FORMALISTIC': 'Formalistic',
        #     'O - OVERALL - NON FORMALISTIC': 'Non-formalistic',
        # }

        return {
            'doc_len_tokens': 'Document length',
            'num_args': "Number of arguments",
            'avg_arg_len': 'Average argument length per document',
            'frac_arg_CL': 'CL fraction',
            'frac_arg_D': 'D fraction',
            'frac_arg_PL': 'PL fraction',
            'frac_arg_HI': 'HI fraction',
            'frac_arg_LIN': 'LIN fraction',
            'frac_arg_PC': 'PC fraction',
            'frac_arg_TI': 'TI fraction',
            'frac_arg_SI': 'SI fraction',
            'O - OVERALL - FORMALISTIC': 'Formalistic',
            'O - OVERALL - NON FORMALISTIC': 'Non-formalistic',
        }

    def set_model(self) -> None:
        """
        Method to set the model
        :return: None
        """
        self.trainer_obj.evaluate_best('test')
        self.model = self.trainer_obj.model

    def get_dataset(self, split: str) -> tuple:
        """
        Method to get the dataset
        :param split: specify which dataset to use
        :return: tuple of scaled dataset (if scaler exists), scaled tabular dataset, labels
        """
        dataset, labels = self.features_obj.get_dataset(split, tabular=True)
        data_numpy = dataset.to_numpy()

        scaled_data = self.scaler.transform(data_numpy) if self.scaler else data_numpy
        tabular_data = self.scaler.transform(dataset) if self.scaler else dataset
        tabular_df = pd.DataFrame(tabular_data, columns=dataset.columns)

        return scaled_data, tabular_df, labels

    def set_explainer(self) -> None:
        """
        Method to set the explainer
        :return: None, but sets the lime and shap explainers
        """
        dataset, tabular_data, labels = self.get_dataset('train')
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            dataset, feature_names=tabular_data.columns, class_names=list(self.features_obj.lab2id.keys()), mode='classification'
        )
        background = torch.FloatTensor(dataset[np.random.choice(dataset.shape[0], size=100, replace=False)])
        self.shap_explainer = shap.DeepExplainer(self.model, background)

    def predict_proba(self, input_data: np.ndarray) -> np.ndarray:
        """
        Method to predict the probability according to the input data
        :param input_data: numpy array of the features
        :return: numpy array of the outputs
        """
        x_tensor = torch.tensor(input_data, dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            outputs = softmax(self.model(x_tensor))
        return outputs.numpy()

    def explain_all_data(self, split: str):
        """
        Method to generalize the explanation process for the LIME
        :param split: string to specify which dataset to use
        :return: None
        """
        dataset, _, _ = self.get_dataset(split)
        for sample in dataset:
            exp = self.lime_explainer.explain_instance(sample, self.predict_proba, num_features=len(sample))
            exp.as_pyplot_figure()
            plt.show()

    def global_explain_sp(self, split: str) -> None:
        """
        Method to generalize the global explanation process for the LIME
        :param split: specifies which dataset to use
        :return: None
        """
        dataset, _, _ = self.get_dataset(split)
        sp_lime_explainer = submodular_pick.SubmodularPick(
            self.lime_explainer,
            dataset,
            self.predict_proba,
            sample_size=dataset.shape[0],
            num_exps_desired=5,
            num_features=dataset.shape[1]
        )

        exp_path = os.path.join(self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}', f'seed_{self.parameters["seed"]}')

        for exp in sp_lime_explainer.sp_explanations:
            label = exp.available_labels()[0]
            fig = exp.as_pyplot_figure(label=label)
            figure_path = os.path.join(exp_path, f'sp_instances_{label}.pdf')
            plt.gcf().set_size_inches(15, 6)
            fig.savefig(figure_path)
            plt.show()

    def average_lime(self, split: str) -> None:
        """
        Method to generalize (by taking the average) the lime explanations
        :param split: string to specify which dataset to use
        :return: None
        """
        dataset, tabular, _ = self.get_dataset(split)
        feature_dict = {feature: list() for feature in tabular.columns}
        for idx in range(dataset.shape[0]):
            exp = self.lime_explainer.explain_instance(dataset[idx], self.predict_proba, num_features=dataset.shape[0], num_samples=500)

            for feature in feature_dict.keys():
                for each in exp.as_list():
                    if feature in each[0]:
                        feature_dict[feature].append(each[1])

        avg_feature = {feat: np.mean(sum(weights)) for feat, weights in feature_dict.items()}

        sorted_features = sorted(avg_feature.items(), key=lambda x: abs(x[1]), reverse=True)
        features, importance_values = zip(*sorted_features)

        plt.figure(figsize=(8, 6))
        bars = plt.barh(features, importance_values, color=['green' if v > 0 else 'red' for v in importance_values])
        plt.xlabel('Average LIME Feature Weight')
        plt.title('Aggregated Global Feature Importances (LIME)')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.show()

    def get_shap_values(self, dataset, save_path):
        if not os.path.exists(save_path):
            shap_values = self.shap_explainer.shap_values(torch.FloatTensor(dataset))
            with open(save_path, 'wb') as f:
                pickle.dump(shap_values, f)
        with open(save_path, 'rb') as f:
            shap_values = pickle.load(f)
        return shap_values

    def shap_explain(self, split: str, save_path: str) -> None:
        """
        Method to provide all relevant shap explanations
        :param split: string to specify which dataset to use
        :return: None
        """
        shap_values_path = os.path.join(save_path, 'shap_values.pickle')

        dataset, tabular, labels = self.get_dataset(split)
        self.scaler = False
        unnorm_ds, unnorm_tabular, _ = self.get_dataset(split)

        reverse_label = {idx: label for label, idx in self.features_obj.lab2id.items()}
        shap_values = self.get_shap_values(dataset, shap_values_path)
        self.compute_dependencies(shap_values, tabular, unnorm_ds, save_path=save_path, reverse_label=reverse_label)


        # exp_path = os.path.join(self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}', f'seed_{self.parameters["seed"]}')
        for label in range(shap_values.shape[-1]):
            feat_names = [each.replace('frac_arg_', '').replace('doc_len_tokens', 'doc. length').replace('avg_arg_len', 'arg. length').replace('num_args', 'num. arg.-s') for each in tabular.columns]

            shap.summary_plot(shap_values[:, :, label], features=torch.FloatTensor(dataset), feature_names=feat_names, max_display=11, show=False)
            figure_path = os.path.join(save_path, f'shap_all_{label}.pdf')
            fig = plt.gcf()
            ax = plt.gca()
            cbar = fig.axes[-1]
            cbar.set_ylabel('Feature Value', fontsize=15)
            cbar.tick_params(labelsize=12)
            # cbar.ax.tick_params(labelsize=40)
            ax.set_title(f'Shap values for {self.report_names[reverse_label[label]]} label', fontsize=17)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            # plt.axis('tight')
            fig.set_size_inches(9, 6)
            fig.subplots_adjust(bottom=0.1, top=0.9)

            plt.savefig(figure_path)

            plt.show()
            plt.close('all')
            shap_exp = shap.Explanation(
                values=shap_values[:, :, label],
                data=unnorm_ds,
                # feature_names=tabular.columns,
                feature_names=feat_names,
            )

            figure_path = os.path.join(save_path, f'shap_bar_{label}.pdf')
            shap.plots.bar(shap_exp, show=False, max_display=11)
            fig = plt.gcf()
            ax = plt.gca()
            fig.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.9)
            fig.set_size_inches(8, 5)
            ax.set_title(f'Mean shap values for {self.report_names[reverse_label[label]]}', fontsize=17)
            plt.savefig(figure_path)
            plt.show()
            plt.close('all')
            for each in ['PL', 'num. arg.-s', 'TI']:
                figure_path = os.path.join(save_path, f'shap_{each}_{label}.pdf')

                shap.plots.scatter(shap_exp[:, each], show=False)
                fig = plt.gcf()
                fig.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.9)
                fig.set_size_inches(8, 5)

                plt.title(f'Shap values of {each} for {self.report_names[reverse_label[label]]} label', fontsize=17)
                plt.yticks(fontsize=14)
                plt.xticks(fontsize=14)
                plt.savefig(figure_path)
                plt.show()
                plt.close('all')

    def compute_dependencies(self, shap_values, tabular, dataset, save_path, reverse_label):
        """
        Method is used to compute dependency between shap values of top 5 features
        :param shap_values: shap values which was computed for the predictions on the test set
        :param split: string to specify which dataset to use
        :param split: string to specify which dataset to use
        :param split: string to specify which dataset to use

        :return: None
        """
        shap_values_array = np.array(shap_values)
        vals = np.abs(shap_values_array).mean(axis=0)
        for label in range(vals.shape[-1]):

            feature_imports = pd.DataFrame(list(zip(tabular.columns, vals[:, label])), columns=['Feature', 'Importance'])
            feature_imports.sort_values('Importance', ascending=False, inplace=True)
            top_features = list(feature_imports['Feature'][:5])
            combinations = list(itertools.combinations(top_features, 2))

            featnames = [self.report_names[each] for each in tabular.columns]
            for idx, combination in enumerate(combinations):
                # fig = plt.figure()
                shap.dependence_plot(
                    self.report_names[combination[0]],
                    shap_values=shap_values[:, :, label],
                    features=dataset,
                    feature_names=featnames,
                    interaction_index=self.report_names[combination[1]],
                    show=False
                )
                main = self.report_names[combination[0]]
                secondary = self.report_names[combination[1]]
                fig = plt.gcf()
                fig.subplots_adjust(left=0.3, right=0.7, bottom=0.1, top=0.9)
                fig.set_size_inches(8, 5)
                plt.tight_layout()
                ax = plt.gca()
                ax.set_title(f"Binary dependency plot for {self.report_names[reverse_label[label]]} label", fontsize=17)
                ax.set_xlabel(f"{main}", fontsize=14)
                ax.set_ylabel(f"SHAP values for {main}", fontsize=14)
                plt.yticks(fontsize=14)
                plt.xticks(fontsize=14)

                fig_name = f'dependence_{combination[0]}_{combination[1]}_label_{label}.pdf'
                fig_path = os.path.join(save_path, fig_name)
                plt.savefig(fig_path)
                plt.close()


    def explain_predictions(self, split):
        self.explanations_path = os.path.join(self.parameters['result_path'], f'exp_{self.parameters["exp_num"]}', f'seed_{self.parameters["seed"]}')
        self.check_dir(self.explanations_path)
        shap_path = os.path.join(self.explanations_path, 'shap')
        self.check_dir(shap_path)
        lime_path = os.path.join(self.explanations_path, 'lime')
        self.check_dir(lime_path)
        self.shap_explain(split=split, save_path=shap_path)