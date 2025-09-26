import numpy as np
from data import DummyDataClassification


class GaussianNaiveBayes:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes, self.classes_counts = np.unique(self.y, return_counts=True)
        self.number_classes = len(self.classes)
        self.priors = None
        self.means = None
        self.stds = None
        self.x_classes_means = None
        self.x_classes_stds = None

    @staticmethod
    def gaussian_pdf(x, mean, std):
        # Add a small value to std to avoid division by zero
        std = std + 1e-9
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    def fit(self):
        self.priors = self.classes_counts / len(self.y)
        self.x_classes_means = np.array([np.mean(self.X[self.y == c], axis=0) for c in range(self.number_classes)])
        self.x_classes_stds = np.array([np.std(self.X[self.y == c], axis=0) for c in range(self.number_classes)])

    def predict(self, X):
        # Initialize an array to store the likelihoods for each class
        likelihoods = np.zeros((X.shape[0], self.number_classes))

        # Calculate the likelihood for each class
        for c in range(self.number_classes):
            # Calculate the PDF for each feature separately
            for i in range(X.shape[1]):  # Iterate over features
                likelihoods[:, c] += np.log(self.gaussian_pdf(X[:, i], self.x_classes_means[c, i], self.x_classes_stds[c, i]))

            # Add the log prior
            likelihoods[:, c] += np.log(self.priors[c])

        # Return the class with the highest likelihood
        return np.argmax(likelihoods, axis=1)

if __name__ == "__main__":
    # X, y = generate_dummy_data(n_samples=300, n_classes=3, random_seed=9092)
    # plot_data(X, y)
    dummy_data_class = DummyDataClassification(n_samples=300,
                                                n_samples_features=2,
                                                n_classes=3,
                                                random_seed=1235,
                                                debug=True)
    data = dummy_data_class.generate_dummy_data()
    dummy_data_class.plot_data(data[0], data[1])
    train_split, valid_split, test_split = dummy_data_class.get_dataset_splits(data)
    dummy_data_class.plot_splits(train_split, valid_split, test_split)
    # classifier instance
    my_gnb_class = GaussianNaiveBayes(X=train_split[0], y=train_split[1])
    # train classifier
    my_gnb_class.fit()
    # valid predictions
    preds_val = my_gnb_class.predict(valid_split[0])
    accuracy_valid = np.mean(preds_val == valid_split[1])
    print(f"Validation Accuracy: {accuracy_valid:.2f}")
    # test predictions
    preds_test = my_gnb_class.predict(test_split[0])
    accuracy_test = np.mean(preds_test == test_split[1])
    print(f"Test Accuracy: {accuracy_test:.2f}")
