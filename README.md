# UEDI: Unsupervised Evaluation of Dataset Integration

Evaluation is a bottleneck in data integration processes: it is performed by domain experts through manual onerous 
data inspections. This task is particularly heavy in real business scenarios, where the large amount of data makes 
checking all integrated tuples infeasible. Our idea is to address this issue by providing the experts with an 
unsupervised measure, based on word frequencies, which quantifies how much a dataset is representative of another 
dataset, giving an indication of how good is the integration process. The paper motivates and introduces the measure 
and provides extensive experimental evaluations, that show the effectiveness and the efficiency of the approach.


For a detailed description of the work please read our [paper](https://dl.acm.org/doi/abs/10.1145/3477314.3507688). 
Please cite the paper if you use the code from this repository in your work.

```
@inproceedings{10.1145/3477314.3507688,
    author = {Paganelli, Matteo and Buono, Francesco Del and Guerra, Francesco and Ferro, Nicola},
    title = {Evaluating the Integration of Datasets},
    year = {2022},
    publisher = {Association for Computing Machinery},
    booktitle = {Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing},
    pages = {347–356},
    series = {SAC '22}
}
```


## Library

### Requirements

- Python: Python 3.*
- Packages: requirements.txt

### Installation

```bash
$ cd source

$ virtualenv -p python3 venv

$ source venv/bin/activate

$ pip install -r requirements.txt
```


Install the necessary datasets/models for nltk functions to work [Example](https://www.nltk.org/install.html#installing-nltk-data)
```python
import nltk
nltk.download()
```

### How to Use

**Input and Output Representativness**

```python
import pandas as pd
from uedi.evaluation import prepare_dataset
from uedi.representativeness import representativness

# Compute input and output representativness
filename = 'data/Structured_Fodors-Zagats.csv'
columns = ['name', 'addr', 'city', 'phone', 'type', 'class']
df = pd.read_csv(filename)
df1, df2, dfi = prepare_dataset(df, columns)

input_repr, output_repr = representativness(df_s=df1, df_i=dfi)
print(f'\nSource 1 Representativness')
print(f'Input Representativness: {input_repr:0.4f}')
print(f'Output Representativness: {output_repr:0.4f}')
```

**Input and Output Ranking Representativness**
```python
import numpy as np
import pandas as pd
from uedi.evaluation import prepare_dataset
from uedi.rank import input_ranking, output_ranking

# Compute input and output representativness
filename = 'data/Structured_Fodors-Zagats.csv'
columns = ['name', 'addr', 'city', 'phone', 'type', 'class']
df = pd.read_csv(filename)
df1, df2, dfi = prepare_dataset(df, columns)

# Compute input ranking
input_ranks = input_ranking(df_s=df1, df_i=dfi)
idx = np.argmin(input_ranks)
print('\nThe least represented record is: ')
print(df1.iloc[idx])


# Compute output ranking
output_ranks = output_ranking(df_list=[df1, df2], df_i=dfi)
idx = np.argmin(output_ranks)
print('\nThe least represented record is: ')
print(dfi.iloc[idx])
```

Please feel free to contact me if you need any further information
