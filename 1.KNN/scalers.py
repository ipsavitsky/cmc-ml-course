import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        min = np.tile(self.min, (data.shape[0], 1))
        max = np.tile(self.max, (data.shape[0], 1))
        return (data - min) / (max - min)


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.E = np.mean(data, axis=0)
        self.D = np.std(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        E = np.tile(self.E, (data.shape[0], 1))
        D = np.tile(self.D, (data.shape[0], 1))
        return (data - E) / D
