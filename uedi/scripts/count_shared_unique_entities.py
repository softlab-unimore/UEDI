from uedi.data_repos.data_manager import DataManager


if __name__ == '__main__':
    repository_id = "deep-matcher"
    dataset_ids = ['Structured_DBLP-GoogleScholar', 'Structured_DBLP-ACM', 'Structured_Amazon-Google',
                   'Structured_Walmart-Amazon', 'Structured_Beer', 'Structured_iTunes-Amazon',
                   'Structured_Fodors-Zagats', 'Textual_Abt-Buy', 'Dirty_iTunes-Amazon', 'Dirty_DBLP-ACM',
                   'Dirty_DBLP-GoogleScholar', 'Dirty_Walmart-Amazon']

    use_cases_map = {'Textual_Abt-Buy': 'U1', 'Dirty_DBLP-ACM': 'U7', 'Structured_iTunes-Amazon': 'U6',
                     'Structured_Amazon-Google': 'U2', 'Structured_DBLP-ACM': 'U8', 'Dirty_DBLP-GoogleScholar': 'U9',
                     'Dirty_Walmart-Amazon': 'U11', 'Dirty_iTunes-Amazon': 'U5', 'Structured_Beer': 'U3',
                     'Structured_DBLP-GoogleScholar': 'U10', 'Structured_Walmart-Amazon': 'U12',
                     'Structured_Fodors-Zagats': 'U4'}

    for dataset_id in dataset_ids:

        print(use_cases_map[dataset_id])

        pre_data_manager = DataManager(repository_id, dataset_id)

        data = pre_data_manager.get_dataset_file("all", "clean")
        num_entities = len(data['entity_id'].unique())
        shared_entities = len(data.groupby('entity_id').filter(lambda x: len(x) > 1)['entity_id'].unique())
        unique_entities = len(data.groupby('entity_id').filter(lambda x: len(x) == 1)['entity_id'].unique())
        print("\tSHARED ENTITIES: {}".format(shared_entities))
        print("\tSHARED ENTITIES (%): {}".format(shared_entities / num_entities))
        print("\tUNIQUE ENTITIES: {}".format(unique_entities))
        print("\tUNIQUE ENTITIES (%): {}".format(unique_entities / num_entities))
