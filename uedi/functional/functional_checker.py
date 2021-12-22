import pandas as pd
import numpy as np
from collections import Counter


def check_fd(fd_list, df, get_violations=False, null_sem=True):
    """
    check how many tuples match the given fd

    :param fd_list: functional dependency in the form of [lhs, rhs] with lhs and rhs two list
    :param df: dataset
    :param get_violations: True if you want to get the list of tuple that violate the FD
    :param null_sem: True if you consider NaN = NaN, False if you consider NaN != NaN
    :return: (right, wrong)
        right= number of tuples that match FD
        wrong= number of tuples that don't match FD
    """
    if not isinstance(fd_list, (list, tuple)):
        raise TypeError("fd_list is not a list or tuple")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df is not a pandas DataFrame")
    if len(fd_list) != 2:
        raise ValueError("fd_list doesn't contain two element")
    if not isinstance(fd_list[0], list) or not isinstance(fd_list[1], list):
        raise ValueError("fd_list doesn't contain two list")

    right = 0
    wrong = 0
    violations = []
    data = df.copy()

    # check another way without convert dataset into string
    data = data.astype(str)

    # get lhs and rhs of FD
    lhs = fd_list[0]
    rhs = fd_list[1]
    data = data[lhs + rhs]

    # get unique lhs values
    lhs_values = data.groupby(lhs).count().index.tolist()

    # for each unique lhs check the unique rhs
    for val in lhs_values:
        rows = data[np.all(data[lhs] == val, axis=1)]
        if 'nan' in val and not null_sem:
            right += len(rows)
            break

        rhs_values = rows.groupby(rhs).count().index.tolist()
        if len(rhs_values) == 1:
            if 'nan' in rhs_values and not null_sem:
                wrong += len(rows)
                if get_violations:
                    violations.append([val, rhs_values])
            else:
                right += len(rows)
        else:
            wrong += len(rows)
            if get_violations:
                violations.append([val, rhs_values])
    if get_violations:
        return violations
    return right, wrong


def check_fd_values(fd, df_integration, df_source):
    # TODO: possible optimization
    # projection of the dataframe on the attributes involved in the functional dependency

    # TODO: possible optimization
    # group by computation for both integration and source datasets
    #
    # S-I true and false scores computation
    # loop over source group by sizes
    # for each fd value, check its existence into the integration group by sizes
    # if the value doesn't exist then single_val_tsi=0 and single_val_fsi=source size
    # otherwise single_val_tsi= source size and single_val_fsi=0
    # tsi = sum(single_val_tsi) and fsi=sum(single_val_fsi)
    #
    # I-S true and false scores computation
    # loop over integration group by size
    # for each fd value, check its existence into the source group by sizes
    # if the value doesn't exist then single_val_tis=0 and single_val_fis=integration size
    # otherwise single_val_tis=min(source size, integration size) and single_val_fis=integration size - single_val_tis
    # tis = sum(single_val_tis) and fis=sum(single_val_fis)

    # get lhs and rhs of FD
    lhs = fd[0]
    rhs = fd[1]

    df_integration = df_integration[lhs + rhs]
    df_source = df_source[lhs + rhs]

    # df_unique_integration = df_integration.drop_duplicates(subset=lhs + rhs).astype(str).values
    # df_unique_source = df_source.drop_duplicates(subset=lhs + rhs).astype(str).values
    #
    # df_integration = df_integration.astype(str).values
    # df_source = df_source.astype(str).values

    df_integration = df_integration.fillna('nan')
    df_source = df_source.fillna('nan')

    df_unique_integration = df_integration.drop_duplicates(subset=lhs + rhs).values
    df_unique_source = df_source.drop_duplicates(subset=lhs + rhs).values

    df_integration = df_integration.values
    df_source = df_source.values

    tis_int = 0
    for fd_val_integration in df_unique_integration:
        for fd_val_source in df_source:
            if np.array_equal(fd_val_integration, fd_val_source):
                tis_int += 1

    tis_source = 0
    for fd_val_source in df_unique_source:
        for fd_val_integration in df_integration:
            if np.array_equal(fd_val_source, fd_val_integration):
                tis_source += 1

    tis = min([tis_int, tis_source])
    fis = len(df_integration) - tis

    res = []
    for fd_val_source in df_source:
        found = False
        for fd_val_integration in df_integration:
            if np.array_equal(fd_val_source, fd_val_integration):
                found = True
                break
        res.append(found)

    res = Counter(res)
    tsi = res.get(True, 0)
    fsi = res.get(False, 0)

    return tis, fis, tsi, fsi


# def check_fd_file(fd_list, file_name, get_violations=False, null_sem=True):
#     """
#     check how many tuples match the given fd
#
#     :param fd_list: functional dependency in the form of [lhs, rhs] with lhs and rhs two list
#     :param file_name: name of the csv dataset
#     :param get_violations: True if you want to get the list of tuple that violate the FD
#     :param null_sem: True if you consider NaN = NaN, False if you consider NaN != NaN
#     :return: (right, wrong)
#         right= number of tuples that match FD
#         wrong= number of tuples that don't match FD
#     """
#     if not isinstance(fd_list, (list, tuple)):
#         raise TypeError("fd_list is not a list or tuple")
#     if not os.path.isfile(file_name):
#         raise TypeError("file_name is not a file")
#     if len(fd_list) != 2:
#         raise ValueError("fd_list doesn't contain two element")
#     if not isinstance(fd_list[0], list) or not isinstance(fd_list[1], list):
#         raise ValueError("fd_list doesn't contain two list")
#
#     right = 0
#     wrong = 0
#     violations = []
#     data = pd.read_csv(file_name)
#
#     # check another way without convert dataset into string
#     data = data.astype(str)
#
#     # get lhs and rhs of FD
#     lhs = fd_list[0]
#     rhs = fd_list[1]
#     data = data[lhs + rhs]
#
#     # get unique lhs values
#     lhs_values = data.groupby(lhs).count().index.tolist()
#
#     # for each unique lhs check the unique rhs
#     for val in lhs_values:
#         rows = data[np.all(data[lhs] == val, axis=1)]
#         if 'nan' in val and not null_sem:
#             right += len(rows)
#             break
#
#         rhs_values = rows.groupby(rhs).count().index.tolist()
#         if len(rhs_values) == 1:
#             if 'nan' in rhs_values and not null_sem:
#                 wrong += len(rows)
#                 if get_violations:
#                     violations.append([val, rhs_values])
#             else:
#                 right += len(rows)
#         else:
#             wrong += len(rows)
#             if get_violations:
#                 violations.append([val, rhs_values])
#     if get_violations:
#         return violations
#     return right, wrong




