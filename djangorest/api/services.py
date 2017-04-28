from random import randint
import random
import json
import csv
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import tree
import pydotplus
import pydot
from subprocess import check_call
from random import seed
from random import randrange
from csv import reader
from math import sqrt

import sys, traceback
from subprocess import check_call


class DataGenerator:
    """
    Generate data.json file
    """
    def exec(self):
        data = []
        for i in range(1, 501):
            if i % 10 == 0:
                data.append({'temperature': float("{0:.1f}".format(random.uniform(35, 36.4))), 'heartbeat': randint(120, 160),
                             'humidity': randint(30, 60), 'sick': 1})
            if i % 10 == 2:
                data.append(
                    {'temperature': float("{0:.1f}".format(random.uniform(37.6, 41))), 'heartbeat': randint(120, 160),
                     'humidity': randint(30, 60), 'sick': 1})
            if i % 10 == 4 or i % 10 == 6:
                data.append({'temperature': float("{0:.1f}".format(random.uniform(36.5, 37.5))), 'heartbeat': randint(40, 120),
                 'humidity': randint(30, 60), 'sick': 1})
            else:
                data.append({'temperature': float("{0:.1f}".format(random.uniform(36.5, 37.5))), 'heartbeat': randint(120, 160),
                 'humidity': randint(30, 60), 'sick': 0})
        with open('../data.json', 'w') as outfile:
            json.dump(data, outfile)
        #print('data ===>'+len(data))

    def convert(self):
        """
        Convert data.json content from json to csv
        and insert it in data.csv
        """
        with open("../data.json") as file:
            json_data = json.load(file)
            fieldnames = ['temperature', 'heartbeat', 'humidity', 'sick']
        with open("../data.csv", "w") as output:
            csv_file = csv.writer(output)
            count = 0
            for item in json_data:
                if count == 0:
                    header = item.keys()
                    csv_file.writerow(header)
                    count += 1
                csv_file.writerow([item['temperature'], item['humidity'], item['heartbeat'], item['sick']])

    def decisionTree(self, entry):
        """
        Run Decision Tree algorithm
        """
        train_df = pd.read_csv("../data.csv")

        y = targets = labels = train_df["sick"].values

        columns = ["temperature", "heartbeat"]
        features = train_df[list(columns)].values

        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        x = imp.fit_transform(features)

        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=4)
        clf = clf.fit(x, y)

        with open("../data2.dot", 'w') as f:
            f = tree.export_graphviz(clf, out_file=f, feature_names=columns)
            check_call(['dot', '-Tpng', '../data.dot', '-o', '../data.png'])

        # add predicitions: when we enter values we get a response
        sick, temperature, heartbeat = entry.items()
        tempK, temp = temperature
        heartK, heart = heartbeat
        return clf.predict([[temp, heart]])
        #print(entry.items())



    # Load a CSV file
    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    # Convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    # Convert string column to integer
    def str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, class_values):
        gini = 0.0
        for class_value in class_values:
            for group in groups:
                size = len(group)
                if size == 0:
                    continue
                proportion = [row[-1] for row in group].count(class_value) / float(size)
                gini += (proportion * (1.0 - proportion))
        return gini

    # Select the best split point for a dataset
    def get_split(self, dataset, n_features):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < n_features:
            index = randrange(len(dataset[0]) - 1)
            if index not in features:
                features.append(index)
        for index in features:
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left, n_features)
            self.split(node['left'], max_depth, min_size, n_features, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth + 1)

    # Build a decision tree
    def build_tree(self, train, max_depth, min_size, n_features):
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root

    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # Create a random subsample from the dataset with replacement
    def subsample(self, dataset, ratio):
        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    # Make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    # Random Forest Algorithm
    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
        trees = list()
        for i in range(n_trees):
            sample = self.subsample(train, sample_size)
            tree = self.build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)
        predictions = [self.bagging_predict(trees, row) for row in test]
        return (predictions)
    def exec(self):
        # Test the random forest algorithm
        seed(1)
        # load and prepare data
        filename = '../data.csv'
        dataset = self.load_csv(filename)
        # convert string attributes to integers
        # for i in range(0, len(dataset[0])-1):
        #	str_column_to_float(dataset, i)
        # convert class column to integers
        # str_column_to_int(dataset, len(dataset[0])-1)
        # evaluate algorithm
        n_folds = 5
        max_depth = 10
        min_size = 1
        sample_size = 1.0
        n_features = int(sqrt(len(dataset[0]) - 1))
        iter = {}
        res = {}
        for n_trees in [1, 5, 10]:
            scores = self.evaluate_algorithm(dataset, self.random_forest, n_folds, max_depth, min_size, sample_size, n_trees,
                                        n_features)
            print('Trees: %d' % n_trees)
            print('Scores: %s' % scores)
            print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
            iter['trees'] = n_trees
            iter['scores'] = scores
            iter['mean_accuracy'] = sum(scores) / float(len(scores))
            res[n_trees] = iter
        return res

