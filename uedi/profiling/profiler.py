import requests
import json
import logging
import time
import os
import pandas as pd
from pathlib import Path

from uedi.profiling.profile_models import *
from uedi.utils.file_utilities import read_multiple_json_objects

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent
logging.basicConfig(level=logging.INFO)


class Profiler(object):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_algorithm(self, dataset, algorithm):
        pass


class MetanomeProfiler(Profiler):

    def __init__(self, metanome_dir, metanome_url):
        Profiler.__init__(self)
        self.metanome_output_dir = os.path.join(metanome_dir, "deployment", "target", "results")
        self.metanome_data_dir = os.path.join(metanome_dir, "deployment", "target", "backend", "WEB-INF", "classes",
                                              "inputData")
        self.metanome_url = metanome_url
        self.metanome_api_url = "{}api/".format(metanome_url)

        self.header = {
            # "Host": "localhost:8081",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "it-IT,it;q=0.8,en-US;q=0.5,en;q=0.3",
            # "Accept-Encoding": "gzip, deflate",
            "Referer": "http://localhost:8080/",
            "Content-Type": "application/json;charset=utf-8",
            # "Content-Length": "1926",
            "Origin": "http://localhost:8080",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }

    def get_profiling_algorithm_id(self, alg_name):

        self.logger.info("Getting id for algorithm {}.".format(alg_name))

        alg_id = None
        url = self.metanome_api_url + "algorithms"
        r = requests.get(url)
        resp_json = json.loads(r.content)
        for alg in resp_json:
            if alg_name in alg["fileName"]:
                alg_id = alg["id"]
                break

        self.logger.debug("{} id = {}.".format(alg_name, alg_id))
        return alg_id

    def get_profiling_algorithm_full_name(self, alg_name):

        self.logger.info("Getting full name for algorithm {}.".format(alg_name))

        alg_full_name = None
        url = self.metanome_api_url + 'algorithms'
        r = requests.get(url)
        resp_json = json.loads(r.content)
        for alg in resp_json:
            if alg_name in alg["fileName"]:
                alg_full_name = alg["fileName"]
                break

        self.logger.debug("{} full name = {}.".format(alg_name, alg_full_name))
        return alg_full_name

    def get_profiling_algorithm_parameters(self, alg_name):

        self.logger.info("Getting parameters for algorithm {}.".format(alg_name))

        alg_full_name = self.get_profiling_algorithm_full_name(alg_name)
        url = self.metanome_api_url + 'parameter/' + alg_full_name
        r = requests.get(url)
        resp_json = json.loads(r.content)
        for item in resp_json:
            del item["fixNumberOfSettings"]

        self.logger.debug("{} parameters = {}.".format(alg_name, resp_json))
        return resp_json

    def get_dataset_info(self, dataset_path):

        clean_dataset_path = dataset_path.split(os.sep)[-1]
        self.logger.info("Getting info for dataset {}.".format(clean_dataset_path))

        dataset_info = None
        url = self.metanome_api_url + 'file-inputs'
        r = requests.get(url)
        resp_json = json.loads(r.content)

        print(resp_json)

        for ds in resp_json:
            if clean_dataset_path == ds["name"]:
                ds["separatorChar"] = ds.pop("separator")
                ds["header"] = ds.pop("hasHeader")
                for item in ["name", "comment"]:
                    del ds[item]
                ds["type"] = "ConfigurationSettingFileInput"
                dataset_info = ds
                break

        self.logger.debug("{} info = {}.".format(clean_dataset_path, dataset_info))
        dataset_info["fileName"] = dataset_path
        return dataset_info

    def _load_dataset(self, dataset):

        self.logger.info("Loading dataset {}.".format(dataset.split(os.sep)[-1]))
        #
        # df = pd.read_csv(os.path.join(ROOT_DIR, dataset))
        # metanome_dataset_path = os.path.join(self.metanome_data_dir, dataset.split(os.sep)[-1])
        # df.to_csv(metanome_dataset_path, index=False)
        #
        # return metanome_dataset_path
        file = os.path.join(ROOT_DIR, dataset)
        url = "{}api/file-inputs/store".format(self.metanome_url)

        self.logger.info("File {0} uploaded using URL {1}".format(file, url))
        m = {'file': open(file, 'rb')}

        r = requests.post(url, files=m)

        metanome_dataset_path = os.path.join(self.metanome_data_dir, dataset.split(os.sep)[-1])

        return metanome_dataset_path

    def _execute_algorithm(self, dataset, alg_name):

        clean_dataset_name = dataset.split(os.sep)[-1]
        self.logger.info("Executing algorithm {} on dataset {}.".format(alg_name, clean_dataset_name))

        url = self.metanome_api_url + 'algorithm-execution/'

        alg_parameters = self.get_profiling_algorithm_parameters(alg_name)

        for alg_parameter in alg_parameters:

            if alg_parameter["type"] == "ConfigurationRequirementRelationalInput":
                alg_parameter["settings"] = [self.get_dataset_info(dataset)]
            else:
                alg_parameter["settings"] = [{"type": alg_parameter["type"].replace("Requirement", "Setting"),
                                              "value": alg_parameter["defaultValues"][0]}]

        clean_dataset_name = clean_dataset_name.split(".")[0]

        body = {
            "algorithmId": self.get_profiling_algorithm_id(alg_name),
            "executionIdentifier": clean_dataset_name + "_" + alg_name + time.strftime("%Y-%m-%d_%H%M%S",
                                                                                              time.gmtime()),
            "requirements": alg_parameters,
            "cacheResults": False,
            "writeResults": True,
            "countResults": False,
            "memory": ""
        }

        r = requests.post(url, headers=self.header, json=body)
        json_response = json.loads(r.content)

        result_file = None
        if "results" in json_response:
            if len(json_response["results"]) > 0:
                if "fileName" in json_response["results"][0]:
                    result_file = json_response["results"][0]["fileName"]

        if result_file:
            result_file_path = os.path.join(self.metanome_output_dir, result_file.split(os.sep)[-1])
        else:
            raise Exception("Result file not generated.")

        self.logger.debug("Algorithm execution completed successfully. Results saved in {}.".format(result_file))

        return result_file_path

    def parse_results(self, result_file):

        self.logger.info("Parsing result file.")

        if "stats" not in result_file:
            result = self.parse_functional_dependecy_results(result_file)

        else:
            result = self.parse_general_statistics(result_file)

        return result

    def parse_ucc_results(self, result_file):

        self.logger.info("Parsing UCC results.")

        with open(result_file, "r") as f:
            columns = {}
            UCCs = []
            results = False
            for line in f:
                if line[0:2] == "1.":
                    columns[(line.split()[0][2:])] = line.split()[1]
                elif line == "# RESULTS\n":
                    results = True
                elif results == True:
                    hs = [int(i) if i else '' for i in line.rstrip("\n").split(",")]
                    hs.sort()
                    UCCs.append(UCC(hs))
        return columns, UCCs

    def parse_functional_dependecy_results(self, result_file):

        self.logger.info("Parsing functional dependency results.")

        dep_type = result_file.split("_")[-1]

        if dep_type == "uccs":
            return self.parse_ucc_results(result_file)

        result_file_content = open(result_file, "r").read()
        deps = []
        for dependency in read_multiple_json_objects(result_file_content):
            dep_lhs = []

            for lhs_col in dependency["determinant"]["columnIdentifiers"]:
                dep_lhs.append(lhs_col["columnIdentifier"])

            dep_rhs = [dependency["dependant"]["columnIdentifier"]]

            deps.append([dep_lhs, dep_rhs])

        return deps

    def parse_general_statistics(self, result_file):

        self.logger.info("Parsing general statistics.")

        with open(result_file, 'r') as myfile:
            data = myfile.read().replace('\n', '')

        ris = pd.DataFrame()
        for obj in read_multiple_json_objects(data):
            tmp = pd.DataFrame.from_dict(obj["statisticMap"])
            tmp["columnIdentifier"] = obj["columnCombination"]["columnIdentifiers"][0]["columnIdentifier"]
            ris = ris.append(tmp)
        ris = ris[ris.index == "value"]
        ris.index = ris["columnIdentifier"]

        return ris

    def execute_algorithm(self, dataset, alg_name):

        self.clean_backend()
        metanome_dataset_path = self._load_dataset(dataset)
        result_file_path = self._execute_algorithm(metanome_dataset_path, alg_name)
        result = self.parse_results(result_file_path)

        return result

    def clean_backend(self):

        url = "{}api/file-inputs".format(self.metanome_url)
        r = requests.get(url)
        r = json.loads(r.content)

        for x in r:
            requests.delete("{}/delete/{}".format(url, x['id']))

        folder = self.metanome_data_dir
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)