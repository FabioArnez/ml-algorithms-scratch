import numpy as np
import matplotlib.pyplot as plt
import math
import random


class DummyData:
    def __init__(self, n_samples, random_seed=1234, debug=False):
        
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.debug = debug

    def generate_dummy_data(self):
        raise NotImplementedError

    def get_dataset_splits(self,
                           dataset,
                           train_size=0.7,
                           valid_size=0.2,
                           test_size=0.1,
                           split_rnd_seed=1234):
        if not math.isclose(train_size + valid_size + test_size, 1.0):
                raise ValueError("train_size, valid_size, and test_size must add up to 1.0")
        
        # get dataset len
        dataset_len = len(dataset[1])
        # gen list with dataset indices
        ds_indices = list(range(dataset_len))
        # indices random shuffle
        random.seed(split_rnd_seed)
        random.shuffle(ds_indices)
        # split indices for train and evaluation
        split_train = int(math.floor(dataset_len * train_size))
        train_idx = ds_indices[:split_train]
        eval_idx = ds_indices[split_train:]
        split_eval = int(math.floor(dataset_len * valid_size))
        valid_idx = eval_idx[:split_eval]
        test_idx = eval_idx[split_eval:]
        if self.debug:
            print("train size: ", len(train_idx))
            print("train data idx:\r\n", train_idx)
            print("eval total size: ", len(eval_idx))
            print("eval data idx:\r\n", eval_idx)
            print("eval valid_idx split size: ", len(valid_idx))
            print("valid data idx:\r\n", valid_idx)
            print("eval test_idx split size: ", len(test_idx))
            print("test data idx:\r\n", test_idx)
        else:
            pass
        # Extract the data using the split indices
        train_split = (dataset[0][train_idx], dataset[1][train_idx])
        valid_split = (dataset[0][valid_idx], dataset[1][valid_idx])
        test_split = (dataset[0][test_idx], dataset[1][test_idx])
        return train_split, valid_split, test_split

    def plot_data(self, X, y):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=60)
        plt.title('2D Dummy Dataset')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Class')
        plt.grid(True)
        plt.show()


class DummyDataClassification(DummyData):
    def __init__(self,
                 n_samples,
                 n_samples_features=2,
                 n_classes=3,
                 random_seed=1234,
                 debug=False):
        super().__init__(n_samples=n_samples, random_seed=random_seed, debug=debug)

        self.n_samples_features = n_samples_features
        self.n_classes = n_classes

    def generate_dummy_data(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        # create placeholders
        X = np.zeros((self.n_samples, self.n_samples_features))
        y = np.zeros(self.n_samples, dtype=int)
        # define the number of samples per class
        samples_class = self.n_samples // self.n_classes
        # generate samples per each class
        for i in range(self.n_classes):
            mean = np.random.uniform(-5, 5, size=self.n_samples_features)
            var = np.random.uniform(0.5, 1.5, size=self.n_samples_features)
            cov = np.diag(var)
            # Generate the same number of samples per class
            samples = np.random.multivariate_normal(mean, cov, size=samples_class)
            X[i * samples_class:(i + 1) * samples_class] = samples
            y[i * samples_class:(i + 1) * samples_class] = i

        return X, y
    
    def plot_splits(self, train_split, valid_split, test_split):   
        X_train, y_train = train_split[0], train_split[1]
        X_valid, y_valid = valid_split[0], valid_split[1]
        X_test, y_test = test_split[0], test_split[1]
        
        # Plot the data splits
        plt.figure(figsize=(15, 5))

        plt.subplot(3, 1, 1)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', s=60)
        plt.title('Training Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Class')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.scatter(X_valid[:, 0], X_valid[:, 1], c=y_valid, cmap='viridis', edgecolors='k', s=60)
        plt.title('Validation Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Class')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k', s=60)
        plt.title('Test Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Class')
        plt.grid(True)
        # plt.tight_layout()
        plt.show()



def generate_dummy_data(n_samples=100,
                        n_samples_features=2,
                        n_classes=3,
                        random_seed=1234,
                        debug=False):
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.zeros((n_samples, n_samples_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_classes):
        mean = np.random.uniform(-5, 5, size=n_samples_features) # choose a rand. unif. val for the mean in this range
        cov = np.diag(np.random.uniform(0.5, 2, size=n_samples_features)) # choose a rand. unif. val for the cov in this range
        samples = np.random.multivariate_normal(mean, cov, size=n_samples // n_classes) # 
        X[i * (n_samples // n_classes):(i + 1) * (n_samples // n_classes)] = samples
        y[i * (n_samples // n_classes):(i + 1) * (n_samples // n_classes)] = i

    return X, y

def get_dataset_splits(dataset,
                       train_size=0.7,
                       valid_size=0.2,
                       test_size=0.1,
                       split_rnd_seed=1234,
                       debug=False):
    if not math.isclose(train_size + valid_size + test_size, 1.0):
            raise ValueError("train_size, valid_size, and test_size must add up to 1.0")
    
    # get dataset len
    dataset_len = len(dataset[1])
    # gen list with dataset indices
    ds_indices = list(range(dataset_len))
    # indices random shuffle
    random.seed(split_rnd_seed)
    random.shuffle(ds_indices)
    # split indices for train and evaluation
    split_train = int(math.floor(dataset_len * train_size))
    train_idx = ds_indices[:split_train]
    eval_idx = ds_indices[split_train:]
    split_eval = int(math.floor(dataset_len * valid_size))
    valid_idx = eval_idx[:split_eval]
    test_idx = eval_idx[split_eval:]
    if debug:
        print("train size: ", len(train_idx))
        print("train data idx:\r\n", train_idx)
        print("eval total size: ", len(eval_idx))
        print("eval data idx:\r\n", eval_idx)
        print("eval valid_idx split size: ", len(valid_idx))
        print("valid data idx:\r\n", valid_idx)
        print("eval test_idx split size: ", len(test_idx))
        print("test data idx:\r\n", test_idx)
    else:
        pass
    # Extract the data using the split indices
    train_split = (dataset[0][train_idx], dataset[1][train_idx])
    valid_split = (dataset[0][valid_idx], dataset[1][valid_idx])
    test_split = (dataset[0][test_idx], dataset[1][test_idx])
    return train_split, valid_split, test_split

def plot_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=60)
    plt.title('2D Dummy Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True)
    plt.show()

def plot_splits(train_split, valid_split, test_split):   
    X_train, y_train = train_split[0], train_split[1]
    X_valid, y_valid = valid_split[0], valid_split[1]
    X_test, y_test = test_split[0], test_split[1]
    
    # Plot the data splits
    plt.figure(figsize=(15, 5))

    plt.subplot(3, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', s=60)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.scatter(X_valid[:, 0], X_valid[:, 1], c=y_valid, cmap='viridis', edgecolors='k', s=60)
    plt.title('Validation Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k', s=60)
    plt.title('Test Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True)

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dummy_data_class = DummyDataClassification(n_samples=300,
                                                n_samples_features=2,
                                                n_classes=3,
                                                random_seed=1235,
                                                debug=True)
    data = dummy_data_class.generate_dummy_data()
    dummy_data_class.plot_data(data[0], data[1])
    train_split, valid_split, test_split = dummy_data_class.get_dataset_splits(data)
    dummy_data_class.plot_splits(train_split, valid_split, test_split)
