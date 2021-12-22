import os
import py_entitymatching as em
import pandas as pd
import collections
import numpy as np
from sklearn.metrics import accuracy_score


class Matcher(object):
    """
    This class implements a matcher.
    """

    def __init__(self, data1, data2, gt, random_state=24):
        """
        This method stores the data to be matched and the ground truth to evaluate the matching logic.

        :param data1: Pandas DataFrame containing the first dataset
        :param data2: Pandas DataFrame containing the second dataset
        :param gt: Pandas DataFrame containing the ground truth
        :param random_state: seed for random choices
        """
        if not isinstance(data1, pd.DataFrame):
            raise TypeError("Wrong data type for the parameter data1. Only Pandas DataFrame data type is allowed.")

        if not isinstance(data2, pd.DataFrame):
            raise TypeError("Wrong data type for the parameter data2. Only Pandas DataFrame data type is allowed.")

        if not isinstance(gt, pd.DataFrame):
            raise TypeError("Wrong data type for the parameter gt. Only Pandas DataFrame data type is allowed.")

        if not isinstance(random_state, int):
            raise TypeError("Wrong data type for the parameter random_state. Only integer data type is allowed.")

        self.data1 = data1
        self.data2 = data2
        self.gt = gt

        self.random_state = random_state

        # generate a set of features
        self.feature_table = em.get_features_for_matching(self.data1, self.data2, validate_inferred_attr_types=False)
        self.train = None
        self.test = None

    def get_data(self):
        """
        This method returns the two original datasets to be integrated.
        :return: two original datasets
        """
        return self.data1, self.data2

    def get_feature_table(self):
        """
        This method retrieves the automatically-generated feature table.
        :return: the feature table
        """

        return self.feature_table

    def set_feature_table(self, feature_table):
        """
        This method sets manually the feature table.
        :return: None
        """

        if not isinstance(feature_table, pd.DataFrame):
            raise TypeError("Wrong data type for parameter feature_table. Only Pandas DataFrame data type is allowed.")

        self.feature_table = feature_table

    def remove_features(self, feature_indexes):
        """
        This method removes features from the automatically-generated feature table.
        :param feature_indexes: the index of the feature to be removed
        :return: a new version of the feature table where the indicated features are filtered out from the table
        """

        if not isinstance(feature_indexes, collections.Iterable):
            raise TypeError("Wrong data type for parameter feature_indexes. Only iterable data type is allowed.")

        for index in feature_indexes:
            if index < 0 or index > len(self.feature_table):
                err_msg = "Wrong data value for single index in the feature_indexes parameter."
                err_msg += " Only values in the range [0,{}] are allowed.".format(len(self.feature_table))
                raise ValueError(err_msg)

        return self.feature_table.drop(feature_indexes)

    def split_train_test(self, ratio):
        """
        This method splits the ground truth data into train and test, based on the user-provided train proportion
        ratio (i.e., ratio parameter).

        :param ratio: the percent of data to be used as train
        :return: ground truth data split in train and test sets
        """

        if not isinstance(ratio, float):
            raise TypeError("Wrong data type for the parameter ratio. Only float data type is allowed.")

        if ratio <= 0 or ratio >= 1:
            raise ValueError("Wrong data value for the parameter ratio. Only values in the range (0, 1) are allowed.")

        train_test = em.split_train_test(self.gt, train_proportion=ratio, random_state=self.random_state)
        self.train = train_test['train']
        self.test = train_test['test']

        return self.train, self.test

    def apply_rule_matcher(self, rules, mode="test"):
        """
        This method creates a rule-based matcher with the user-provided rules and applies it to the test data. Based
        on the selected modality (i.e., the mode parameter), different test sets are considered (e.g., the whole ground
        truth, only the test set or only the train set).

        :param rules: the rules of the rule-based matcher
        :param mode: the test modality
        :return: the dataset integrated by the user-provided rules and the evaluation scores
        """

        if not isinstance(rules, collections.Iterable):
            raise TypeError("Wrong data type for the parameter rules. Only iterable data type is allowed.")

        for rule in rules:
            if not isinstance(rule, collections.Iterable):
                raise TypeError("Wrong data type for a single rule. Only iterable data type is allowed.")

            for cond in rule:
                if not isinstance(cond, str):
                    raise TypeError("Wrong data type for a single rule condition. Only string data type is allowed.")

        if not isinstance(mode, str):
            raise TypeError("Wrong data type for the parameter mode. Only string data type is allowed.")

        modes = ["all", "train", "test"]
        if mode not in modes:
            raise ValueError("Wrong data value for the parameter mode. Only values {} are allowed.".format(modes))

        # create a boolean rule matcher
        brm = em.BooleanRuleMatcher()

        # add rules to boolean rule matcher
        for rule in rules:
            brm.add_rule(rule, self.feature_table)

        # predict on the test set
        test_data = None
        if mode == "all":
            test_data = self.gt
        elif mode == "train":
            test_data = self.train
        elif mode == "test":
            test_data = self.test

        predictions = brm.predict(test_data, target_attr='pred_label', append=True)

        # evaluate predictions
        eval_result = em.eval_matches(predictions, 'label', 'pred_label')
        eval_result["accuracy"] = accuracy_score(predictions['label'], predictions['pred_label'])

        return predictions, eval_result

    def apply_multi_rule_matchers(self, multi_matcher_rules, mode="test"):
        """
        This method applies multiple rules and creates multiple matching pairs.

        :param multi_matcher_rules: the rules to be applied in order to create multiple matching pairs
        :param mode: test modality
        :return: multiple matching pairs data with the related effectiveness scores
        """

        if not isinstance(multi_matcher_rules, collections.Iterable):
            raise TypeError("Wrong data type for parameter multi_matcher_rules. Only iterable data type is allowed.")

        for matcher_rules in multi_matcher_rules:
            if not isinstance(matcher_rules, collections.Iterable):
                raise TypeError("Wrong data type for multi_matcher_rules elements. Only iterable data type is allowed.")

            for matcher_rule in matcher_rules:
                if not isinstance(matcher_rule, collections.Iterable):
                    raise TypeError(
                        "Wrong data type for single rule in multi_matcher_rules. Only iterable data type is allowed.")

                for cond in matcher_rule:
                    if not isinstance(cond, str):
                        raise TypeError(
                            "Wrong data type for single cond in multi_matcher_rules. Only string data type is allowed.")

        modes = ["all", "train", "test"]
        if mode not in modes:
            raise ValueError("Wrong data value for the parameter mode. Only values {} are allowed.".format(modes))

        results = []
        for matcher_rules in multi_matcher_rules:
            match_data, match_eval = self.apply_rule_matcher(matcher_rules, mode=mode)

            result = [match_data.copy(), match_eval]
            results.append(result)

        return results

    def apply_ml_matcher(self, model_name):
        """
        This method generates some matching features for training a ML model and applies it to the test data.

        :param model_name: the name of the ML model to be used
        :return: the dataset integrated by the ML model and the evaluation scores
        """

        # check parameter data types
        if not isinstance(model_name, str):
            raise TypeError("Wrong data type for parameter model_name. Only string data type is allowed.")

        # check parameter data values
        models = ['DecisionTree', 'SVM', 'RF', 'LogReg', 'LinReg', 'All']
        if model_name not in models:
            raise ValueError("Wrong data value for parameter model_name. Only values {} are allowed.".format(models))

        # featurize train and test data
        train_feature_set = em.extract_feature_vecs(self.train, feature_table=self.feature_table, attrs_after=['label'],
                                                    show_progress=False)

        test_feature_set = em.extract_feature_vecs(self.test, feature_table=self.feature_table, attrs_after=['label'],
                                                   show_progress=False)

        def _fill_nan_in_feature_vectors(feature_vectors):

            # Check if the feature vectors contain missing values
            # A return value of True means that there are missing values
            null_presence = pd.isnull(feature_vectors).sum().sum()

            if null_presence:
                # Impute feature vectors with the mean of the column values.
                feature_vectors = em.impute_table(feature_vectors, missing_val=np.nan,
                                                  exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                                  strategy='mean')

            return feature_vectors

        train_feature_set = _fill_nan_in_feature_vectors(train_feature_set)
        test_feature_set = _fill_nan_in_feature_vectors(test_feature_set)

        # model selection
        model = None
        if model_name == "DecisionTree":
            model = em.DTMatcher(name=model_name, random_state=self.random_state)
        elif model_name == "SVM":
            model = em.SVMMatcher(name=model_name, random_state=self.random_state)
        elif model_name == "RF":
            model = em.RFMatcher(name=model_name, random_state=self.random_state)
        elif model_name == "LogReg":
            model = em.LogRegMatcher(name=model_name, random_state=self.random_state)
        elif model_name == "LinReg":
            model = em.LinRegMatcher(name=model_name)
        elif model_name == "All":
            dt = em.DTMatcher(name='DecisionTree', random_state=self.random_state)
            svm = em.SVMMatcher(name='SVM', random_state=self.random_state)
            rf = em.RFMatcher(name='RF', random_state=self.random_state)
            lg = em.LogRegMatcher(name='LogReg', random_state=self.random_state)
            ln = em.LinRegMatcher(name='LinReg')

            matchers = [dt, svm, rf, lg, ln]
            result = em.select_matcher(matchers, table=train_feature_set,
                                       exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                       k=5,
                                       target_attr='label', metric_to_select_matcher='f1',
                                       random_state=self.random_state)

            best_matcher_id = result['cv_stats']["Average f1"].idxmax()

            model = matchers[best_matcher_id]

        model.fit(table=train_feature_set, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                  target_attr='label')

        predictions = model.predict(table=test_feature_set, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                    target_attr='pred_label', append=True, inplace=True)

        eval_result = em.eval_matches(predictions, 'label', 'pred_label')
        eval_result["accuracy"] = accuracy_score(predictions['label'], predictions['pred_label'])

        return predictions, eval_result

    def apply_ml_matcher_with_different_features(self, model_name, features_confs):
        """
        This method applies a ML matcher multiple times with different feature configurations. Together with the
        user-provided feature configurations, the configuration that considers all the features will be also tested.

        :param model_name: the name of the ML matcher
        :param features_confs: the list of feature configurations
        :return: multiple matching pairs data with the related effectiveness scores
        """
        # check parameter data types
        if not isinstance(model_name, str):
            raise TypeError("Wrong data type for parameter model_name. Only string data type is allowed.")

        if not isinstance(features_confs, collections.Iterable):
            raise TypeError("Wrong data type for parameter features_confs. Only iterable data type is allowed.")

        for feature_conf in features_confs:
            if not isinstance(feature_conf, collections.Iterable):
                raise TypeError("Wrong data type for single conf in features_confs. Only iterable data type is allowed.")

        # check parameter data values
        models = ['DecisionTree', 'SVM', 'RF', 'LogReg', 'LinReg', 'All']
        if model_name not in models:
            raise ValueError("Wrong data value for parameter model_name. Only values {} are allowed.".format(models))

        # all feature configuration
        match_data, match_eval = self.apply_ml_matcher(model_name)
        all_features_result = [match_data.copy(), match_eval]

        results = [all_features_result]
        for feature_conf in features_confs:
            small_feature_table = self.remove_features(feature_conf)
            self.set_feature_table(small_feature_table)
            match_data, match_eval = self.apply_ml_matcher(model_name)

            result = [match_data.copy(), match_eval]
            results.append(result)

        return results


if __name__ == '__main__':
    # Get the datasets directory
    datasets_dir = em.get_install_path() + os.sep + 'datasets'

    path_A = datasets_dir + os.sep + 'dblp_demo.csv'
    path_B = datasets_dir + os.sep + 'acm_demo.csv'
    path_labeled_data = datasets_dir + os.sep + 'labeled_data_demo.csv'

    A = em.read_csv_metadata(path_A, key='id')
    B = em.read_csv_metadata(path_B, key='id')

    # Load the pre-labeled data
    S = em.read_csv_metadata(path_labeled_data,
                             key='_id',
                             ltable=A, rtable=B,
                             fk_ltable='ltable_id', fk_rtable='rtable_id')
    random_seed = 24

    matcher = Matcher(A, B, S, random_state=random_seed)
    train_proportion = 0.5
    matcher.split_train_test(train_proportion)

    # RULE MATCHER
    rules = [
        ['title_title_lev_sim(ltuple, rtuple) > 0.4', 'year_year_exm(ltuple, rtuple) == 1'],
        ['authors_authors_lev_sim(ltuple, rtuple) > 0.4']
    ]
    match_data, match_eval = matcher.apply_rule_matcher(rules, mode="all")

    # # ML MATCHER
    # ml_model_name = "DecisionTree"
    # # ml_model_name = "SVM"
    # # ml_model_name = "RF"
    # # ml_model_name = "LogReg"
    # # ml_model_name = "LinReg"
    # # ml_model_name = "All"
    # match_data, match_eval = matcher.apply_ml_matcher(ml_model_name)

    print(match_data)

    em.print_eval_summary(match_eval)
