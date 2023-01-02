import pandas as pd
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error
from math import sqrt


def check_parameter_type(parameter, parameter_name, parameter_type, type_name, optional_param=False):
    """
    This function checks whether the data type of the input parameter corresponds with the provided data type.

    :param parameter: parameter whose data type has to be checked
    :param parameter_name: name of the parameter to be checked
    :param parameter_type: the right data type of the parameter
    :param type_name: data type string
    :param optional_param: boolean flag that indicates whether the input parameter could be None
    :return: None
    """

    check = True
    if optional_param:
        if parameter is None:
            check = False

    if check:
        if not isinstance(parameter, parameter_type):
            print(type(parameter))
            raise TypeError("Wrong data type for {}. Only {} data type is allowed.".format(parameter_name, type_name))


def check_cols_in_dataframe(df: pd.DataFrame, columns: list):
    """
    This function checks whether the provided columns are included in the input Pandas DataFrame object.

    :param df: Pandas DataFrame object
    :param columns: list of column names
    :return: None
    """

    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(columns, 'columns', list, 'list')

    for col in columns:
        if col not in df.columns.values:
            raise ValueError("Column {} not found.".format(col))


def get_binary_classification_effectiveness_report(y_true: list, y_pred: list, flat: bool = True):
    """
    This function creates a report that evaluates the effectiveness of the input predictions (y_pred) with related to
    the input ground truth (y_true).

    :param y_true: true labels
    :param y_pred: predicted labels
    :param flat: boolean flag that indicates whether the report format has to be flat
    :return: dictionary containing the effectiveness report
    """

    check_parameter_type(y_true, 'y_true', list, 'list')
    check_parameter_type(y_pred, 'y_pred', list, 'list')
    check_parameter_type(flat, 'flat', bool, 'boolean')

    eval_res = classification_report(y_true, y_pred, output_dict=True)

    if not flat:
        return eval_res

    res_class_0 = eval_res['0']
    res_class_1 = eval_res['1']
    prec0 = res_class_0['precision']
    rec0 = res_class_0['recall']
    f10 = res_class_0['f1-score']
    support0 = res_class_0['support']
    prec1 = res_class_1['precision']
    rec1 = res_class_1['recall']
    f11 = res_class_1['f1-score']
    support1 = res_class_1['support']
    acc = eval_res['accuracy']

    report_data = {
        'prec0': prec0,
        'rec0': rec0,
        'f10': f10,
        'support0': support0,
        'prec1': prec1,
        'rec1': rec1,
        'f11': f11,
        'support1': support1,
        'acc': acc
    }

    return report_data


def get_regression_metric_scores(y_test, y_pred):
    metric_scores = {}
    # R squared
    r2score = r2_score(y_test, y_pred)
    # mean absolute error
    absolute_loss = mean_absolute_error(y_test, y_pred)
    # mean squared error
    squared_loss = mean_squared_error(y_test, y_pred)
    # root mean squared error
    rms_loss = sqrt(mean_squared_error(y_test, y_pred))

    metric_scores["r2_score"] = r2score
    metric_scores["absolute_loss"] = absolute_loss
    metric_scores["squared_loss"] = squared_loss
    metric_scores["rms_loss"] = rms_loss

    return metric_scores
