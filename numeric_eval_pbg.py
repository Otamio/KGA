import logging

logger = logging.getLogger("torchbiggraph")

import pandas as pd
import faiss, json
from torchbiggraph.operators import ComplexDiagonalDynamicOperator, TranslationDynamicOperator, DiagonalDynamicOperator
import numpy as np
from functools import lru_cache
import h5py, torch
from glob import glob


class NumericMetricsReporter(object):
    def __init__(self, checkpoint_path, index_type='Flat'):

        _, dataset, model = checkpoint_path.split('/')
        self._dataset = dataset
        self._model = model
        self._index_type = index_type
        self._entity_path = f"numeric/{dataset}/edge/entities"
        self._model_path = f"numeric/{dataset}/{model}"

        print("Entity File:", self._entity_path)
        print("Model File:", self._model_path)

        # Entity to index mapping
        with open(f'{self._entity_path}/entity_names_all_0.json') as fd:
            self._entity_list = json.load(fd)
        self._entity_index = dict()
        for i, qnode in enumerate(self._entity_list):
            self._entity_index[qnode] = i

        self._entity_count = len(self._entity_index)
        self._index_entity = {v: k for k, v in self._entity_index.items()}

        # relational to index mapping
        with open(f'{self._entity_path}/dynamic_rel_names.json') as fd:
            self._rel_type_names = json.load(fd)
        self._rel_index = {r: j for j, r in enumerate(self._rel_type_names)}
        self._index_rel = {v: k for k, v in self._rel_index.items()}

        # operators
        self._n_properties = len(self._rel_type_names)

        medians_file = f"numeric/{dataset}/medians.dict"
        print("Median File:", medians_file)
        with open(medians_file) as fd:
            self._medians = json.load(fd)

        self.load_eval_files()

    def load_eval_files(self):
        self._valid = pd.read_csv(f'numeric/{self._dataset}/valid_raw.txt', sep='\t', header=None)
        self._valid[0] = self._valid[0].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
        self._valid[1] = self._valid[1].apply(lambda x: x if not "com" in x else x.split("com")[1][:-1])
        self._valid[1] = self._valid[1].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
        self._valid.columns = ['node1', 'label', 'node2']

        self._test = pd.read_csv(f'numeric/{self._dataset}/test_raw.txt', sep='\t', header=None)
        self._test[0] = self._test[0].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
        self._test[1] = self._test[1].apply(lambda x: x if not "com" in x else x.split("com")[1][:-1])
        self._test[1] = self._test[1].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
        self._test.columns = ['node1', 'label', 'node2']

    def update_index(self):

        embedding_file = glob(f"{self._model_path}/embeddings_all_0.v*.h5")[0]
        model_file = glob(f"{self._model_path}/model.v*.h5")[0]

        # Load the embeddings
        with h5py.File(embedding_file, "r") as hf:
            self._embedding = torch.from_numpy(hf["embeddings"][...])
        # Parse the dimensions
        self._dim = self._embedding.shape[1]
        # Fetch the FAISS index
        if self._model in ['ComplEx', 'DistMult']:
            self._index = faiss.index_factory(self._dim, self._index_type, faiss.METRIC_INNER_PRODUCT)
        else:
            self._index = faiss.index_factory(self._dim, self._index_type, faiss.METRIC_L2)
        self._index.train(self._embedding.detach().numpy())
        self._index.add(self._embedding.detach().numpy())

        # Load the operator's state dict (relation embeddings)
        if 'complex' in self._model_path:

            self._operator_lhs = ComplexDiagonalDynamicOperator(self._dim, self._n_properties)
            self._operator_rhs = ComplexDiagonalDynamicOperator(self._dim, self._n_properties)

            with h5py.File(model_file, "r") as hf:
                operator_state_dict_lhs = {
                    "real": torch.from_numpy(hf["model/relations/0/operator/lhs/real"][...]),
                    "imag": torch.from_numpy(hf["model/relations/0/operator/lhs/imag"][...]),
                }
                operator_state_dict_rhs = {
                    "real": torch.from_numpy(hf["model/relations/0/operator/rhs/real"][...]),
                    "imag": torch.from_numpy(hf["model/relations/0/operator/rhs/imag"][...]),
                }
        elif 'transe' in self._model_path:

            self._operator_lhs = TranslationDynamicOperator(self._dim, self._n_properties)
            self._operator_rhs = TranslationDynamicOperator(self._dim, self._n_properties)

            with h5py.File(model_file, "r") as hf:
                operator_state_dict_lhs = {
                    "translations": torch.from_numpy(hf["model/relations/0/operator/lhs/translations"][...]),
                }
                operator_state_dict_rhs = {
                    "translations": torch.from_numpy(hf["model/relations/0/operator/rhs/translations"][...]),
                }
        elif 'distmult' in self._model_path:

            self._operator_lhs = DiagonalDynamicOperator(self._dim, self._n_properties)
            self._operator_rhs = DiagonalDynamicOperator(self._dim, self._n_properties)

            with h5py.File(model_file, "r") as hf:
                operator_state_dict_lhs = {
                    "diagonals": torch.from_numpy(hf["model/relations/0/operator/lhs/diagonals"][...]),
                }
                operator_state_dict_rhs = {
                    "diagonals": torch.from_numpy(hf["model/relations/0/operator/rhs/diagonals"][...]),
                }
        else:
            raise NotImplementedError

        self._operator_lhs.load_state_dict(operator_state_dict_lhs)
        self._operator_rhs.load_state_dict(operator_state_dict_rhs)

        self.get_embed.cache_clear()
        self._neighbors_faiss.cache_clear()

    @lru_cache(maxsize=50000)
    def get_embed(self, head, relation=None):
        ''' This function generate the embeddings for the tail entities:
                Head entities: Obtained from the model
                Head + relation: Obtained using torch
        '''
        if relation is None:
            return self._embedding[self._entity_index[head], :].detach().numpy()
        return self._operator_lhs(
            self._embedding[self._entity_index[head], :].view(1, self._dim),
            torch.tensor([self._rel_index[relation]])
        ).detach().numpy()[0]

    @lru_cache(maxsize=50000)
    def _neighbors_faiss(self, head, relation, tail=None, k=10):
        ''' This function returns the nearest neighbors
            given the head and the relation (Using the Faiss index)
        '''
        if not tail:
            return self._index.search(self.get_embed(head, relation).reshape(1, -1), k)
        else:
            return self._index.search(self.get_embed_rev(tail, relation).reshape(1, -1), k)

    def get_neighbors(self, head, relation, tail=None, k=10):
        if not head in self._entity_index:
            return None, None
        scores, ranking = self._neighbors_faiss(head, relation, tail, k=k)
        top_entities = [self._index_entity[index] for index in ranking[0] if index > 0]
        top_scores = scores[0][:len(top_entities)]
        return top_scores, top_entities

    @property
    def index(self):
        return self._index

    def model_pred(self, row):

        label = row['label']
        intervals_left = row['interval_left']
        intervals_right = row['interval_right']

        if intervals_left is None and intervals_right is None:
            return None

        left = None
        for interval in intervals_left:
            if label in interval:
                left = self._medians[interval]

        right = None
        for interval in intervals_right:
            if label in interval:
                right = self._medians[interval]

        if left is None and right is None:
            return self._medians[label]
        elif left is None:
            return right
        elif right is None:
            return left
        return (left + right) / 2

    def report(self):

        # Validation file
        self._valid.loc[:, 'interval_left'] = self._valid.apply(
            lambda r: self.get_neighbors(r['node1'], 'Interval-' + r['label'] + '_left', k=1000)[1], axis=1)
        self._valid.loc[:, 'interval_right'] = self._valid.apply(
            lambda r: self.get_neighbors(r['node1'], 'Interval-' + r['label'] + '_right', k=1000)[1], axis=1)

        self._valid['model'] = self._valid.apply(self.model_pred, axis=1)
        self._valid['error'] = abs(self._valid['model'] - self._valid['node2'])

        # Report the statistics
        logger.info(
            f"/valid/global \t MAE: {self._valid['error'].mean()} \t RMSE: {np.sqrt(np.square(self._valid['error']).mean())}")
        for label in self._valid['label'].unique():
            sli = self._valid[self._valid['label'] == label]
            logger.info(f"{label} \t MAE: {sli['error'].mean()} \t RMSE: {np.sqrt(np.square(sli['error']).mean())}")

        # Test file
        self._test.loc[:, 'interval_left'] = self._test.apply(
            lambda r: self.get_neighbors(r['node1'], 'Interval-' + r['label'] + '_left', k=1000)[1], axis=1)
        self._test.loc[:, 'interval_right'] = self._test.apply(
            lambda r: self.get_neighbors(r['node1'], 'Interval-' + r['label'] + '_right', k=1000)[1], axis=1)
        self._test['model'] = self._test.apply(self.model_pred, axis=1)
        self._test['error'] = abs(self._test['model'] - self._test['node2'])

        # Report the statistics
        logger.info(
            f"/test/global \t MAE: {self._test['error'].mean()} \t RMSE: {np.sqrt(np.square(self._test['error']).mean())}")
        for label in self._test['label'].unique():
            sli = self._test[self._test['label'] == label]
            logger.info(f"{label} \t MAE: {sli['error'].mean()} \t RMSE: {np.sqrt(np.square(sli['error']).mean())}")
