import logging
import os

from uedi.profiling.profiler import MetanomeProfiler


class DataProfilerManager(object):

    def __init__(self, profiler):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.profiler = profiler

    def execute_profiling_algorithms_multi_dataset(self, datasets, algorithms):

        self.logger.info("Executing algorithms {} on datasets {}.".format(algorithms, datasets))

        dataset_profiling_map = {}

        for dataset in datasets:

            dataset_key = dataset.split(os.sep)[-1]
            results = self.execute_profiling_algorithms(dataset, algorithms)

            dataset_profiling_map[dataset_key] = results

        return dataset_profiling_map

    def execute_profiling_algorithms(self, dataset, algorithms):

        results = []
        for algorithm in algorithms:
            result = self.profiler.execute_algorithm(dataset, algorithm)
            results.append(result)

        return results


if __name__ == "__main__":


    metanome_dir = "/home/matteo/Scaricati/metanome_repo/Metanome"
    metanome_url = "http://127.0.0.1:8081/"
    metanome_profiler = MetanomeProfiler(metanome_dir, metanome_url)

    #datasets = ["data/source1.csv", "data/source2.csv", "data/source3.csv", "data/integration.csv"]
    datasets = ["data/input/source3.csv"]
    algorithms = ["HyFD"]

    profiler_manager = DataProfilerManager(metanome_profiler)
    results = profiler_manager.execute_profiling_algorithms_multi_dataset(datasets, algorithms)
    print(results)
    exit(1)
