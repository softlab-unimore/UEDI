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

    # 1 way: iterative integration
    for i, source in enumerate(sources):
        print(i, source)
        df = pd.read_csv(source)

        # integration phase
        if df_integration is not None:
            # update the new integration dataset
            block_attribute = "Citta"
            rules = [
                ["Nome_Nome_exm(ltuple, rtuple) == 1", "Cognome_Cognome_exm(ltuple, rtuple) == 1",
                 "Citta_Citta_exm(ltuple, rtuple) == 1"],
            ]

            integrator = TwoSourcesIntegrator()
            df_integration = integrator.apply_rule_based_integration(
                "data/integration/integrated_dataset_{}.csv".format(i - 1), source, block_attribute, rules)
        else:
            df_integration = df

        # save new integration dataset
        integration_file_name = "data/integration/integrated_dataset_{}.csv".format(i)
        df_integration.to_csv(integration_file_name, index=False)

        # get functional dependencies both for current data source and integrated dataset
        algorithms = ["HyFD"]

        # compute functional dependencies
        fd_source_all_algorithms = profiler_manager.execute_profiling_algorithms(source, algorithms)
        fd_integration_all_algorithms = profiler_manager.execute_profiling_algorithms(integration_file_name, algorithms)
        fd_source = fd_source_all_algorithms[0]
        fd_integration = fd_integration_all_algorithms[0]

        # df_integration = pd.read_csv("data/integration/integration.csv")
        # fd_source = [[["Citta"], ["Cap"]]]
        # fd_integration = [[["Nome"], ["Cognome"]]]

        # update fd_list with the new function dependency
        for fd in fd_source:
            fd = FunctionalDependency(lhs=fd[0], rhs=fd[1])
            if fd not in fd_list:
                fd_list.append(fd)

        for fd in fd_integration:
            fd = FunctionalDependency(lhs=fd[0], rhs=fd[1])
            if fd not in fd_list:
                fd_list.append(fd)

        # compute the new point for each fd
        for fd_model in fd_list:
            ts, fs = check_fd(fd_model.get(), df)
            ti, fi = check_fd(fd_model.get(), df_integration)
            # ti, fi, ts, fs = check_fd_values(fd_model.get(), df_integration=df_integration, df_source=df)
            fd_model.update(idx=i, ti=ti, fi=fi, ts=ts, fs=fs)

    # Visualize fd history
    for fd_model in fd_list:
        print(fd_model)
        print(fd_model.get_acc_points()[0])
        print(fd_model.get_acc_points()[1])
        print(fd_model.get_acc_cum_points())
        fd_model.plot()


if __name__ == '__main__':
    main()