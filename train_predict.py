#!/usr/bin/env python
import pandas as pd
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score


def main():
    args = parse_args()
    modeler = Modeler(args)
    modeler.create_training_data()
    modeler.train_model()
    modeler.write_predictions()


def parse_args():
    description = 'Train and predict species presence values'
    parser = ArgumentParser(description=description)
    parser.add_argument('train', help='location of training csv')
    parser.add_argument('grid', help='location of grid csv for predictions')
    parser.add_argument('outfile', help='csv to fill with presence predictions')
    parser.add_argument('--n_estimators',
                        help='number of trees in the random forest', type=int,
                        default = 10)
    return parser.parse_args()


class Modeler(object):
    def __init__(self, args):
        self.args = args

    def create_training_data(self):
        data = pd.read_csv(self.args.train)
        self.column_names = list(data.columns.values)

        self._get_user_input()

        self.target = data[self.target_column_name]
        self.predictors = data[self.feature_column_names]
        print "\nTarget values:"
        print self.target.head()
        print "\nPredictor values:"
        print self.predictors.head()

    def _get_user_input(self):
        #self.target_column_name = 'NAME'
        prompt = 'Enter your target variable'
        self.target_column_name = read_column_names(prompt, self.column_names)

        self.column_names.remove(self.target_column_name)

        #self.feature_column_names = ['197902_CHN', '197903_HCC', '197904_MCC',
        #                        '197907_SWH', '197907_WSP', '197912_SST']
        prompt = 'Enter your predictor variables (comma separated)'
        self.feature_column_names = read_column_names(prompt, self.column_names)

    def train_model(self):
        imputed_predictors = impute_predictors(self.predictors)
        self.model = RandomForestClassifier(n_estimators = self.args.n_estimators)
        self.model = self.model.fit(imputed_predictors, self.target)
        print "\nModel:"
        print self.model
        score = cross_val_score(self.model, imputed_predictors, self.target).mean()
        print "\nCV score:", score

    def write_predictions(self):
        grid = pd.read_csv(self.args.grid)
        predictors = grid[self.feature_column_names]
        imputed_predictors = impute_predictors(predictors)
        target_probabilities = self.model.predict_proba(imputed_predictors)[:,1]
        grid[self.target_column_name] = target_probabilities
        grid.to_csv(self.args.outfile, index=False)


def read_column_names(text, column_names):
    input_str = raw_input(text + ': ' + ', '.join(column_names) + '\n')
    input_str = input_str.replace(' ', '')
    input_names = input_str.split(',')
    for name in input_names:
        assert(name in column_names)
    if len(input_names) == 1:
        return input_names[0]
    return input_names


def impute_predictors(predictors):
    imputer = Imputer(strategy="mean")
    imputer = imputer.fit(predictors)
    return imputer.transform(predictors)


if __name__ == '__main__':
    main()
