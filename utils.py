#-------------------------- Tensorflow implementation of MIL for MSI data ---------------------------#

# version 1.1 - February 2024
# Created by WIL GARDNER
# Centre for Materials and Surface Science
# La Trobe University

#----------------------------------------------------------------------------------------------------#

import os, h5py
import numpy as np
import tensorflow as tf

#CLASSES
#Cosine annealing learning rate
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_learning_rate, warmup_epochs, total_epochs, steps_per_epoch):
    
        self.init_learning_rate = init_learning_rate
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = steps_per_epoch * warmup_epochs
        self.decay_steps = steps_per_epoch * (total_epochs - warmup_epochs)

    def __call__(self, step):
        learning_rate = tf.where(
            step < self.warmup_steps,
            self.init_learning_rate * (step / self.warmup_steps),
            self.init_learning_rate * 0.5 * (1 + tf.math.cos(
                np.pi * tf.cast((step - self.warmup_steps), tf.float32) / self.decay_steps))
        )
        return learning_rate

    def get_config(self):
        return {
            'init_learning_rate': self.init_learning_rate,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'steps_per_epoch': self.steps_per_epoch,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#Data generator for training
class DataGen(tf.keras.utils.Sequence):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.steps_per_epoch = len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = np.zeros((1, 1))
        y[:, 0] = self.y[index]    
        return X, y
    
    def __len__(self):
        return self.steps_per_epoch

#FUNCTIONS
#Data importers
def load_mat_file(path):
    #Only usable with .mat files saved with version 7.3. Must be ONLY a single data array in file
    f = h5py.File(path)
    for v in f.values():
        X = np.array(v).astype('float32')
        new_axes = ()
        for i in reversed(range(len(X.shape))):
            new_axes = new_axes + (i,)
        X = np.transpose(X, axes=new_axes)
    f.close()
    return X

#Load data from paths
def load_data_from_paths(image_paths, file_format='.mat'):
    X = []
    X_shapes = []
    
    # Determine the total number of images
    n_images = len(image_paths)

    # Load images
    for i, image_path in enumerate(image_paths):
        if file_format == '.mat':
            image = load_mat_file(image_path)
        elif file_format == '.npy':
            image = np.load(image_path)
        else:
            raise ValueError("Unsupported file format. Please use '.mat' or '.npy'.")
        #TODO add other import options
        X_shapes.append(image.shape)
        X.append(np.reshape(image, (-1, image.shape[-1])))
        if (i+1) % 10 == 0:
            print('{} of {} images loaded.'.format(i+1, n_images))
    return X, X_shapes

#Train test split from path
def path_train_test_split(path, train_size, seed=None, file_format='.mat'):
    if file_format not in ['.mat', '.npy']:
        raise ValueError("Unsupported file format. Please use '.mat' or '.npy'.")
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    train_paths, test_paths, train_labels, test_labels = [], [], [], []
    for class_id, folder in enumerate(subfolders):
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(file_format)]
        # Split the data
        n = len(image_files)
        if seed is not None:
            rand_perm = np.random.RandomState(seed=seed).permutation(n)
        else:
            rand_perm = np.random.permutation(n)
        train_ind = rand_perm[:round(train_size*n)]
        for i, p in enumerate(rand_perm):
            f = image_files[p]
            if i < len(train_ind):
                train_paths.append(f)
                train_labels.append(class_id)
            else:
                test_paths.append(f)
                test_labels.append(class_id)
    return np.array(train_paths), np.array(test_paths), np.array(train_labels), np.array(test_labels)

#Normalise rows by sum
def row_norm(X):
    X_sums = []
    for i in range(len(X)):
        X_sum = np.sum(X[i], axis=1, keepdims=True)
        X_sum[X_sum==0] = 1 #Avoid division by zero
        X[i] /= X_sum
        X_sums.append(X_sum)
    return X, X_sums

#Compute min and max of list of arrays
def compute_minmax(X):
    minmax_list = [(np.min(x), np.max(x)) for x in X]
    X_min = min(v[0] for v in minmax_list)
    X_max = max(v[1] for v in minmax_list)
    return X_min, X_max

#Scale data with min and max
def scale_data(X, X_min, X_max):
    for i in range(len(X)):
        X[i] = (X[i]-X_min)/(X_max-X_min)
    return X