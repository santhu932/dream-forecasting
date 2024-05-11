import numpy as np
import csv
import math
import torch
import torchvision.transforms as transforms
from warnings import warn
import numpy.ma as ma
import os
from torch.utils.data import Subset

# INPUT_DIR = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/ncep_WP_EP_new_2'
INPUT_DIR = '/N/project/hurricane-deep-learning/data/ncep_extracted_41x161'
CSV_FILE = INPUT_DIR + '/tc_0h.csv'

CHANNELS = {
    'ugrdprs-800': 0,
    'ugrdprs-200': 1,
    'vgrdprs-800': 2,
    'vgrdprs-200': 3,
    'vvelprs-500': 4,
    'tmpprs-900': 5,
    'tmpprs-500': 6,
    'rhprs-750': 7,
    'hgtprs-500': 8,
    'absvprs-900': 9,
    'absvprs-700': 10,
    'pressfc': 11,
    'capesfc': 12,
    'tmpsfc': 13,
    'landmask': 14
}

# ----------------------load data operations ----------------------
def load_data(cfg, num_channels=14, combine=True, normalize=None, change_initial = False):
    '''
    Load data from npz file
    :return: a numpy array of shape (N, C, 41, 161) that represents the features
    '''
    all_data = np.load('/N/project/uq-dl/tc_earthformer/data/all.npy')
    if change_initial:
        print("Changing Initial Conditions")
        torch.manual_seed(cfg.seed)
        gaussian_noise = np.random.normal(cfg.mean_noise, cfg.std_noise, size=(all_data.shape[0], all_data.shape[1], all_data.shape[2], all_data.shape[3]))
        all_data = all_data + cfg.noise_factor * gaussian_noise
        
    if num_channels == 14:
        all_data = all_data[:, :14]
    elif num_channels == 6:
        # only select uu, vv, vo
        all_data = all_data[:, [0, 1, 2, 3, 9, 10]]
    elif num_channels == 3:
        all_data = all_data[:, [0, 2, 9]]
    if normalize:
        all_data = normalize(all_data)
    if combine:
        return all_data
    else:
        # seperate all data into chunks by year
        chunks = []
        for i in range(0, all_data.shape[0], 856):
            chunks.append(all_data[i:i + 856])
        return chunks


def round_timestamp(t):
    # corner cases
    if t[11:13] =='03':
        return t[:11] + '00:00:00'
    elif t[11:13] =='21':
        return t[:11] + '18:00:00'
    elif t[11:13] =='09':
        return t[:11] + '06:00:00'
    elif t[11:13] =='15':
        return t[:11] + '12:00:00'
    else:
        return t


def load_tcg_locations():
    '''
    Load labels from npz file
    :param lag: integers, 1 represents 6h, 2 represents 12h, etc.
    :return: a numpy array of shape (N,) that represents the labels
    '''
    tcg_csv = '/N/project/uq-dl/tc_earthformer/data/tcg.csv'
    if os.path.exists(tcg_csv):
        with open(tcg_csv, 'r') as file:
            reader = list(csv.reader(file))
            tcg_locations = []
            for row in reader[1:]:
                tcg_locations.append(eval(row[2]))
            return tcg_locations
    else:
        last_time_stamp = None
        with open(CSV_FILE, 'r') as file:
            reader = list(csv.reader(file))
            N = len(reader) - 1
            offset = (5, 100)
            locations = []
            timestamps = []
            indices = []
            end_timestamps = []
            end_indices = []
            for i in range(N):
                row = reader[i + 1]
                if last_time_stamp is None or last_time_stamp != row[1]:
                    last_time_stamp = row[1]
                    timestamps.append(row[1])
                    indices.append(i)
                    if eval(row[2]):
                        locations.append([(eval(row[9]) - offset[0], eval(row[8]) - offset[1])])
                        end_timestamps.append([row[11]])
                    else:
                        locations.append([])
                        end_timestamps.append([])
                else:
                    locations[-1].append((eval(row[9]) - offset[0], eval(row[8]) - offset[1]))
                    end_timestamps[-1].append(row[11])
            for i in range(len(end_timestamps)):
                try:
                    end_indices.append([timestamps.index(round_timestamp(t)) for t in end_timestamps[i]])
                except ValueError:
                    print('end timestamp not found', i, end_timestamps[i])
                    end_indices.append([])
        with open(tcg_csv, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'time', 'location', 'end_index', 'end_time'])
            for i in range(len(locations)):
                writer.writerow([indices[i], timestamps[i], locations[i], end_indices[i], end_timestamps[i]])
        return locations


def load_tc_locations():
    '''
    Load labels from npz file
    :param lag: integers, 1 represents 6h, 2 represents 12h, etc.
    :return: a numpy array of shape (N,) that represents the labels
    '''
    tc_csv = '/N/project/uq-dl/tc_earthformer/data/tc.csv'
    if os.path.exists(tc_csv):
        with open(tc_csv, 'r') as file:
            reader = list(csv.reader(file))
            locations = []
            for row in reader[1:]:
                locations.append(eval(row[2]))
            return locations
    else:
        last_time_stamp = None
        timestamps = []
        with open(CSV_FILE, 'r') as file:
            reader = list(csv.reader(file))
            N = len(reader) - 1
            locations = []
            offset = (5, 100)
            for i in range(N):
                row = reader[i + 1]
                if last_time_stamp is None or last_time_stamp != row[1]:
                    last_time_stamp = row[1]
                    timestamps.append(row[1])
                    locations.append(eval(row[5]))
                else:
                    for (lat, long) in eval(row[5]):
                        if (lat, long) not in locations[-1]:
                            locations[-1].append((lat, long))
            tcg_locations = load_tcg_locations()
            new_locations = []
            for t in range(len(locations)):
                loc = locations[t]
                tcg_loc = tcg_locations[t]
                new_loc = []
                for (lat, long) in loc:
                    if (lat - offset[0], long - offset[1]) not in tcg_loc:
                        new_loc.append((lat - offset[0], long - offset[1]))
                    else:
                        # pass
                        print('TC location is in TCG location', t, (lat, long))
                new_locations.append(new_loc)
            with open(tc_csv, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['index', 'time', 'location'])
                for i in range(len(new_locations)):
                    writer.writerow([i, timestamps[i], new_locations[i]])
            return new_locations


def load_binary_labels(existing=False, combine=True):
    if existing:
        locations = load_tc_locations()
    else:
        locations = load_tcg_locations()
    binary_labels = [(1 if len(loc) > 0 else 0) for loc in locations]
    if combine:
        return np.array(binary_labels)
    else:
        chunks = []
        for i in range(0, len(binary_labels), 856):
            chunks.append(np.array(binary_labels[i:i + 856]))
        return chunks


def load_labels(label_type='binary', existing=False, gaussian_radius=0, clip_prob=0.1, smooth=True, combine=True):
    '''
    Load labels from npz file
    :param label_type: 'binary' or 'grid'
    :param lag: integers, 1 represents 6h, 2 represents 12h, etc.
    :param existing: if True, we mark existing TCs as 1 also otherwise we only mark generatings as 1.
    :param gaussian_radius: the radius to compute the gaussian probabilites around a TC.
    :param clip_prob: the threshold to clip the gaussian probabilities.
    :return: a numpy array of shape (N,) that represents the labels
    '''
    if label_type == 'binary':
        warn('This is deprecated, use function "load_binary_labels" instead', DeprecationWarning, stacklevel=2)
        return load_binary_labels(existing, combine)
    else:
        if existing:
            locations = load_tc_locations()
        else:
            locations = load_tcg_locations()
        targets = np.zeros((len(locations), 41, 161))
        for i in range(len(locations)):
            for (lat, long) in locations[i]:
                dim0 = math.floor(lat)
                dim1 = math.floor(long)
                if gaussian_radius == 0:
                    targets[i, dim0, dim1] = 1
                else:
                    x = np.arange(0, 41, 1)
                    y = np.arange(0, 161, 1)
                    xx, yy = np.meshgrid(x, y, indexing='ij')
                    gaussian_probs = np.exp(-((xx - dim0) ** 2 + (yy - dim1) ** 2) / (2 * gaussian_radius ** 2))
                    gaussian_probs[gaussian_probs < clip_prob] = 0
                    targets[i] += gaussian_probs
                if not smooth:
                    targets[i] = np.where(targets[i] > 0, 1, 0)
        if combine:
            return targets
        else:
            chunks = []
            for i in range(0, len(locations), 856):
                chunks.append(targets[i:i + 856])
            return chunks


# ----------------------normalization operations ----------------------
def zscore(train, test=None):
    '''
    Normalize data by subtracting mean and dividing by standard deviation
    :param train: training data
    :param test: testing data
    :return: normalized training data and testing data
    '''
    mean = np.mean(train, axis=(0, 2, 3), keepdims=True)
    std = np.std(train, axis=(0, 2, 3), keepdims=True)
    train = (train - mean) / std
    if test is not None:
        test = (test - mean) / std
        return train, test
    else:
        return train


def minmax(train, test=None):
    '''
    Normalize data by subtracting mean and dividing by standard deviation
    :param train: training data
    :param test: testing data
    :return: normalized training data and testing data
    '''
    minimum = np.min(train, axis=(0, 2, 3), keepdims=True)
    maximum = np.max(train, axis=(0, 2, 3), keepdims=True)
    train = (train - minimum) / (maximum - minimum)
    if test is not None:
        test = (test - minimum) / (maximum - minimum)
        return train, test
    else:
        return train


def masked_minmax(train, test=None):
    """
    Normalize data between 0 and 1, but mask the data on land.
    """
    # For the original landmask, 0 means ocean and 1 means land, we need to reverse it.
    landmask = 1 - np.load('/N/project/uq-dl/tc_earthformer/data/landmask.npy')
    landmask = np.tile(landmask, (train.shape[0], train.shape[1], 1, 1)).astype(bool)
    masked_data = ma.masked_array(train, mask=landmask)
    minimum = ma.min(masked_data, axis=(0, 2, 3), keepdims=True)
    maximum = ma.max(masked_data, axis=(0, 2, 3), keepdims=True)
    normalized_train = (train - minimum) / (maximum - minimum)
    # filtering out the extreme values, notice that after min max, the extreme values lies out of [0, 1]
    normalized_train = np.where(normalized_train > 1., 1., normalized_train)
    normalized_train = np.where(normalized_train < 0., 0., normalized_train)
    if test is not None:
        normalized_test = (test - minimum) / (maximum - minimum)
        normalized_test = np.where(normalized_test > 1., 1., normalized_test)
        normalized_test = np.where(normalized_test < 0., 0., normalized_test)
        return normalized_train, normalized_test
    else:
        return normalized_train

# ----------------------split data operations ----------------------
def split(data, labels, train_ratio=0.8):
    '''
    Split data into training and testing set
    :param data: data to be split
    :param labels: labels to be split
    :param train_ratio: ratio of training data
    :param shuffle: if True, shuffle the data before splitting
    :return: training data, training labels, testing data, testing labels
    '''
    if isinstance(data, list):
        N = len(data)
        num_chunks_train = int(N * train_ratio)
        train_data = np.concatenate(data[:num_chunks_train], axis=0)
        train_labels = np.concatenate(labels[:num_chunks_train], axis=0)
        test_data = np.concatenate(data[num_chunks_train:], axis=0)
        test_labels = np.concatenate(labels[num_chunks_train:], axis=0)
        return train_data, train_labels, test_data, test_labels
    else:
        N = data.shape[0]
        train_size = int(N * train_ratio)
        train_data = data[:train_size]
        train_labels = labels[:train_size]
        test_data = data[train_size:]
        test_labels = labels[train_size:]
        return train_data, train_labels, test_data, test_labels


def temporal_data(T_in, T_out, data, labels=None):
    '''
    Convert data into temporal data inside one chunk.
    :param T_in: number of input time steps
    :param T_out: number of output time steps
    :param data: data to be converted
    :param labels: labels to be converted
    :return: temporal data and labels
    '''
    if labels is None:
        labels = data
    X = []
    Y = []
    N = data.shape[0]
    if T_out == 0:
        overlap = 1  # nowcasting
    else:
        overlap = 0
    for i in range(N - T_in - T_out + overlap + 1):
        X.append(data[i:(i + T_in)])
        Y.append(labels[(i + T_in - overlap):(i + T_in + T_out - overlap)])
    return np.stack(X, axis=0), np.stack(Y, axis=0)


def expand_positive(data, labels, flip_dims=(-1, -2, (-1, -2))):
    # apply flip to positive samples. Make sure that the last two dimensions are height and width.
    # return only positive data
    new_data = []
    new_labels = []
    for i in range(data.shape[0]):
        if np.any(labels[i, 0] > 0):
            new_data.append(data[i])
            new_labels.append(labels[i])
            for flip_dim in flip_dims:
                new_data.append(np.flip(data[i], axis=flip_dim))
                if labels.ndim > 2:
                    new_labels.append(np.flip(labels[i], axis=flip_dim))
                else:
                    new_labels.append(labels[i])
    return np.array(new_data), np.array(new_labels)


def reduce_negative(data, labels, num_negatives):
    if labels.ndim > 2:
        binary_labels = np.array([1 if np.any(label > 0) else 0 for label in labels])
    else:
        binary_labels = labels
    negative_indices = np.where(binary_labels == 0)[0]
    num_negatives = min(num_negatives, negative_indices.shape[0])
    chosen_neg_indices = np.random.choice(negative_indices, num_negatives, replace=False)
    return data[chosen_neg_indices], labels[chosen_neg_indices]


def balance_data(data, labels, ratio):
    # get all positive data and flip them
    positive_data, positive_labels = expand_positive(data, labels)
    num_negatives = int(positive_data.shape[0] * ratio)
    negative_data, negative_labels = reduce_negative(data, labels, num_negatives)
    return np.concatenate((positive_data, negative_data)), np.concatenate((positive_labels, negative_labels))

def filter_csv_indices(csv_file, remove_repeat=False):
    tcg_csv = '/N/project/uq-dl/tc_earthformer/data/tcg.csv'
    with open(tcg_csv, 'r') as file:
        reader = list(csv.reader(file))
        tcg_timestamps = [row[1] for row in reader[1:]]
        del file, reader
    with open(csv_file, 'r') as file:
        reader = list(csv.reader(file))
        indices = []
        for row in reader[1:]:
            timestamp = row[0]
            indices.append(tcg_timestamps.index(timestamp))
        del file, reader
        
    if remove_repeat:
        indices = np.unique(indices)
    else:
        indices = np.array(indices)
    return indices


def filter_csv(dataset, csv_file, remove_repeat=False):
    """
    Filter the dataset by the csv file. Notice that the first column of the csv file must be the timestamp of predicted target.
    """
    indices = filter_csv_indices(csv_file, remove_repeat)
    filtered_indices = [i for i in range(len(dataset)) if dataset[i][0].item() in indices]
    return Subset(dataset, filtered_indices)


# ----------------------pytorch operations ----------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, indices, transform=None):
        self.data = data
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.indices[idx]
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return index, sample, label


def get_dataloader(data, labels, start_index=0, batch_size=32, shuffle=False, transform=None):
    if start_index is None:
        indices = -torch.ones(data.shape[0])
    else:
        indices = torch.arange(start_index, start_index + data.shape[0])
    if transform is None:
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels)
        dataset = CustomDataset(data, labels, indices)
    else:
        dataset = CustomDataset(data, labels, indices, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    data = load_data()
    labels = load_labels(label_type='grid', gaussian_radius=0, clip_prob=0.1, smooth=True)
    X, Y = temporal_data(4, 1, data)
    print(X.shape, Y.shape)
    # print(np.max(data[:, 11]), np.min(data[:, 11]))
    # landmask = 1 - np.load('/N/project/uq-dl/tc_earthformer/data/landmask.npy')
    # landmask = np.tile(landmask, (data.shape[0], 1, 1)).astype(bool)
    # masked_data = ma.masked_array(data[:, 11], mask=landmask)
    # minimum = ma.min(masked_data)
    # maximum = ma.max(masked_data)
    # print(maximum, minimum)
    # land_data = ma.masked_array(data[:, 11], mask=1 - landmask)
    # minimum = ma.min(land_data)
    # maximum = ma.max(land_data)
    # print(maximum, minimum)
    # print(data.shape)
    # tcg_labels = load_tcg_locations()
    # print(len(tcg_labels))
    # tc_labels = load_tc_locations()
    # print(len(tc_labels))
    # data = np.load('/N/project/uq-dl/tc_earthformer/data/landmask.npy')
    # data = np.tile(data, (2, 1, 1, 1))
    # print(data.shape)
    # normalized = masked_minmax(data)
    # print(np.sum(normalized))
    # print(masked_minmax(data))
