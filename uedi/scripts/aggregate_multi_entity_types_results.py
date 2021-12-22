import pandas as pd
import os
from os import listdir
from os.path import isfile, join, abspath

results_type = 'multi-entity-types'
# results_type = 'multi-match-percentages'

results_dir = join(abspath(''), 'data', 'output', 'results', results_type)

results_files = [join(results_dir, f) for f in listdir(results_dir) if
                 isfile(join(results_dir, f)) and f.endswith('.csv') and 'aggregated' not in f]

for f in results_files:
    print(f)
    df = pd.read_csv(f)

    relative_file_name = f.split(os.sep)[-1]
    relative_file_name_parts = relative_file_name.split('_')
    first_part = '_'.join(relative_file_name_parts[:-1])
    last_part = relative_file_name_parts[-1]
    out_f = "{}_aggregated_{}".format(first_part, last_part)

    grouped_data = df.groupby(['scenario', 'mode', 'ngram', 'embed_manager', 'data_type']).mean().reset_index()
    grouped_data.to_csv(join(results_dir, out_f))
