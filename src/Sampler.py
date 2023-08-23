import pandas as pd
import numpy as np
import torch
from typing import List
from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import RobustScaler


from sklearn.preprocessing import MinMaxScaler

def coreset_pytorch(data, percentage, sampler):
    """
    Creates a coreset for the given data using the specified sampler and percentage of samples.

    Args:
    - data: numpy array of shape (N, D) representing N data points in D-dimensional space
    - percentage: float between 0 and 1 representing the percentage of data points to be selected for the coreset
    - sampler: instance of a coreset sampler class with a _compute_greedy_coreset_indices method that takes in
              a numpy array of shape (N, D) representing data and returns a numpy array of shape (M,) representing
              the indices of the selected samples

    Returns:
    - numpy array of shape (M, D) representing M data points in D-dimensional space, where M is percentage*N
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.from_numpy(data).to(device)
    sampler_instance = sampler(percentage=percentage, device=device)
    coreset_indices = sampler_instance._compute_greedy_coreset_indices(features)
    return data[coreset_indices, :]






import abc
from typing import Union
import tqdm
import logging
import numpy as np
import torch



class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            logging.error('[BaseSampler] Percentage value not in (0, 1)')
            raise ValueError('Percentage value not in (0, 1)')
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        logging.info('[GreedyCoresetSampler] Initializing sampler...')

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        logging.info('[GreedyCoresetSampler] Applying sparse random projection...')
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.
        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.
        Args:
            features: [NxD] input feature bank to sample.
        """
        logging.info('[GreedyCoresetSampler] Selecting greedy coreset...')
        
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.
        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.
        Args:
            features: [NxD] input feature bank to sample.
        """
        logging.info('[ApproximateGreedyCoresetSampler] Selecting greedy coreset samples...')
        
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.
        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
    
    
    



def generate_training_corsets(Training_df,cols_feature,list_features_group,percentage = 0.1):
    
    list_corsets=[]
    scaler_list=[]
    X_train_scaler=Training_df[cols_feature].to_numpy()[:int(len(Training_df)*0.1)]
    
    for features_group in list_features_group:
        scaler=MinMaxScaler()
        scaler.fit(X_train_scaler[:,features_group])
        scaler_list.append(scaler)
    
    
    
    
    for i in range(len(list_features_group)):
        scaler=scaler_list[i]
        x_train=Training_df[cols_feature].to_numpy()
        x_train=x_train[int(len(Training_df)*0.1):,list_features_group[i]]
        x_train_tranform=scaler.transform(x_train)
         

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        sampler = ApproximateGreedyCoresetSampler

        # Specify percentage of data points to select

        # Generate coreset
        coreset_data = coreset_pytorch(data=x_train_tranform, percentage=percentage, sampler=sampler)
        list_corsets.append(coreset_data)
    return(scaler_list,list_corsets)

def generate_training_corsets_real_data(X_train,list_features_group,percentage = 0.1):
    
    list_corsets=[]

    if percentage==1:
        for i in range(len(list_features_group)):
            X_train_feat_id=X_train[:,list_features_group[i]]
            list_corsets.append(X_train_feat_id)


   
    else:
        
        for i in range(len(list_features_group)):
            X_train_feat_id=X_train[:,list_features_group[i]]


            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            sampler = ApproximateGreedyCoresetSampler

            # Specify percentage of data points to select

            # Generate coreset
            coreset_data = coreset_pytorch(data=X_train_feat_id, percentage=percentage, sampler=sampler)
            list_corsets.append(coreset_data)
    return(list_corsets)
