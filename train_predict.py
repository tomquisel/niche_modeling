#!/usr/bin/env python
import pandas as pd
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score

def main():
    args = parse_args()
    target, predictors, target_name, feature_column_names = create_training_data(args)
    model = train_model(target, predictors, args)
    write_predictions(model, target_name, feature_column_names, args)

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

def create_training_data(args):
    data = pd.read_csv(args.train)
    column_names = list(data.columns.values)
    target_column_name = raw_input('Enter your target variable: ' + ', '.join(column_names) + '\n')
    #target_column_name = 'NAME'
    column_names.remove(target_column_name)
    feature_column_names = raw_input('Enter your predictor variables (comma separated): ' + ', '.join(column_names) + '\n')
    #feature_column_names = '197902_CHN, 197903_HCC, 197904_MCC, 197907_SWH, 197907_WSP, 197912_SST'
    feature_column_names = feature_column_names.replace(' ', '')

    target = data[target_column_name]
    predictors = data[feature_column_names.split(',')]
    print target.head()
    print predictors.head()
    return target, predictors, target_column_name, feature_column_names.split(',')


def train_model(target, predictors, args):
    imputer = Imputer(strategy="mean")
    imputer = imputer.fit(predictors)
    imputed_predictors = imputer.transform(predictors)
    model = RandomForestClassifier(n_estimators = args.n_estimators)
    model = model.fit(imputed_predictors, target)
    print model
    score = cross_val_score(model, imputed_predictors, target).mean()
    print "CV score:", score
    return model

def write_predictions(model, target_name, feature_column_names, args):
    grid = pd.read_csv(args.grid)
    predictors = grid[feature_column_names]
    imputer = Imputer(strategy="mean")
    imputer = imputer.fit(predictors)
    imputed_predictors = imputer.transform(predictors)
    grid[target_name] = model.predict_proba(imputed_predictors)[:,1]
    grid.to_csv(args.outfile, index=False)

if __name__ == '__main__':
    main()
