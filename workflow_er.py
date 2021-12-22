import os
import pandas as pd
from uedi.utils import get_files
from uedi.functional.functional_dependecy import FunctionalDependency
from uedi.integration import TwoSourcesIntegrator
from uedi.profiling import MetanomeProfiler
from uedi.profiling import DataProfilerManager
from uedi.functional.functional_checker import check_fd, check_fd_values


def main():
    sources = get_files(os.path.join("data", "input"))

    # integration dataset
    df_integration = None

    # functional dependencies list
    fd_list = []

    # profiler initialization
    metanome_dir = "/home/matteo/Scaricati/metanome_repo/Metanome"
    metanome_url = "http://127.0.0.1:8081/"
    metanome_profiler = MetanomeProfiler(metanome_dir, metanome_url)
    profiler_manager = DataProfilerManager(metanome_profiler)

    # iterative em
    case = '1'
    change = '1'
    sources = ["data/input/new/data1.csv", "data/input/new/data2.csv", "data/input/new/data3_case{}.csv".format(case)]
    integrations = ["data/input/new/data1.csv", "data/input/new/integration_1_2.csv",
                    "data/input/new/integration_3_{}_{}.csv".format(case, change)]

    for i, source in enumerate(sources):
        print(i, source)
        df = pd.read_csv(source)

        if i == 0:
            continue

        # integration phase
        # if df_integration is not None:
        #     # update the new integration dataset
        #     block_attribute = "Citta"
        #     rules = [
        #         ["Nome_Nome_exm(ltuple, rtuple) == 1", "Cognome_Cognome_exm(ltuple, rtuple) == 1",
        #          "Citta_Citta_exm(ltuple, rtuple) == 1"],
        #     ]
        #
        #     integrator = TwoSourcesIntegrator()
        #     df_integration = integrator.apply_rule_based_integration(
        #         "data/integration/integrated_dataset_{}.csv".format(i - 1), source, block_attribute, rules)
        # else:
        #     df_integration = df

        # save new integration dataset
        # the file must be saved to perform fd discovery using metanome
        # integration_file_name = "data/integration/integrated_dataset_{}.csv".format(i)
        # df_integration.to_csv(integration_file_name, index=False)

        # get functional dependencies both for current data source and integrated dataset
        # algorithms = ["HyFD"]

        # compute functional dependencies
        # fd_source_all_algorithms = profiler_manager.execute_profiling_algorithms(source, algorithms)
        # fd_integration_all_algorithms = profiler_manager.execute_profiling_algorithms(integration_file_name, algorithms)
        # fd_source = fd_source_all_algorithms[0]
        # fd_integration = fd_integration_all_algorithms[0]

        # manual entity resolution
        df_integration = pd.read_csv(integrations[i])
        # fd_source = [[["Country"], ["City"]]]
        fd_source = [[["City"], ["Zip"]]]
        fd_integration = []

        # update fd_list with the new function dependency
        for fd in fd_source + fd_integration:
            fd = FunctionalDependency(lhs=fd[0], rhs=fd[1])
            if fd not in fd_list:
                fd_list.append(fd)

        # compute the new precision and accuracy for each fd
        for fd_model in fd_list:
            # compute precision for each previous source
            matches = []
            for si in sources:
                # match = (ti, fi, ts, fs)
                ti, fi, ts, fs = check_fd_values(fd_model.get(), df_integration=df_integration, df_source=pd.read_csv(si))
                match = {
                    'ti': ti,
                    'fi': fi,
                    'ts': ts,
                    'fs': fs,
                }
                matches.append(match)
                if si == source:
                    break
            # update fd with the new precision value
            fd_model.update_with_history(idx=i, matches=matches)

            # compute accuracy in the current iteration and update
            ts, fs = check_fd(fd_model.get(), df)
            ti, fi = check_fd(fd_model.get(), df_integration)
            fd_model.update_acc(idx=i, ti=ti, fi=fi, ts=ts, fs=fs)

    # Visualize fd history
    for fd_model in fd_list:
        print(fd_model)
        # fd_model.plot_acc()

        print("macro points")
        print(fd_model.get_macro()[0])
        print(fd_model.get_macro()[1])
        print()
        print("micro points")
        print(fd_model.get_micro()[0])
        print(fd_model.get_micro()[1])
        print()
        fd_model.plot_macro()
        fd_model.plot_micro()


if __name__ == '__main__':
    main()