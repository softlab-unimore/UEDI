import pandas as pd
import numpy as np
import os
from uedi.data_integration.data_preparation import DataPreparationComponent
from uedi.models.data_models import IntegratedDataset


class DataFusionComponent(object):
    """
    This class manages the data fusion process for an integrated dataset.
    """

    def __init__(self, integrated_dataset):
        """
        This method stores the user-provided integrated dataset and check if the it needs to be fused by checking the
        existence of entities referenced by multiple records.

        :param integrated_dataset: an IntegratedDataset object to be fused
        """

        if not isinstance(integrated_dataset, IntegratedDataset):
            raise TypeError(
                "Wrong data type for parameter integrated_dataset. Only IntegratedDataset data type is allowed.")

        self.integrated_dataset = integrated_dataset
        self.data = self.integrated_dataset.get_data()
        self.id_col = self.integrated_dataset.get_id_col()
        self.entity_column = self.integrated_dataset.get_entity_label_col()
        self.source_column = self.integrated_dataset.get_source_id_col()
        self.entity_counts = self.data[self.entity_column].value_counts()
        self.entity_ids = self.entity_counts.index.values

        # distinguish between entities that are referenced by single or multiple records
        # entities referenced by multiple records have to be resolved
        # entities referenced by a single record can be emitted in output without any transformation
        entity_multiple_records_ids = self.entity_counts[self.entity_counts > 1].index.values
        entity_single_record_ids = self.entity_counts[self.entity_counts == 1].index.values

        entity_multiple_records = self.data[self.data[self.entity_column].isin(entity_multiple_records_ids)]
        if len(entity_multiple_records) == 0:
            print("Data fusion not needed: all entities are identified by a single record.")

        entity_single_record = self.data[self.data[self.entity_column].isin(entity_single_record_ids)]

        self.data_for_fusion = entity_multiple_records
        self.entity_ids_for_fusion = entity_multiple_records_ids
        self.single_entities = entity_single_record
        self.single_entity_ids = entity_single_record_ids

    def select_records_by_source_id(self, source_id):
        """
        This method implements a data fusion technique by selecting, among the records the refer to the same entity, the
        first record deriving from the user-provided source_id.

        :param source_id: the source identifier with which to select a single record among conflicting records
        :return: a new IntegratedDataset object where for each entity only one record is kept
        """

        # FIXME: as already done in the method select_random_records, add a new column in the merged dataset that
        # FIXME: indicates if each entity was a match or a non-match

        if len(self.data[self.data[self.source_column] == source_id]) == 0:
            raise ValueError("Source identifier {} doesn't exist.".format(source_id))

        merged_entities_ids = self.data_for_fusion.groupby(self.entity_column) \
            .apply(lambda x: x.where(x[self.source_column] == source_id).first_valid_index())
        merged_entities = self.data_for_fusion.loc[merged_entities_ids]

        # check to have considered all entities: it is possible that all records referencing the same entity derive from
        # a data source different from source_id. these entities will be ignored in the previous code, so this is the
        # moment to integrate them inside the final result
        # for these entities the first record is selected
        entities_multiple_records = self.data_for_fusion[self.entity_column].unique()
        entities_multiple_records_resolved = merged_entities[self.entity_column].unique()
        if len(entities_multiple_records) != len(entities_multiple_records_resolved):
            if len(entities_multiple_records_resolved) > len(entities_multiple_records):
                raise Exception(
                    "The data fusion task has created a number of multi-record entities greater than the original one.")

            rem_entities_multiple_records = set(entities_multiple_records).difference(
                entities_multiple_records_resolved)
            rem_data = self.data_for_fusion[
                self.data_for_fusion[self.entity_column].isin(rem_entities_multiple_records)]
            rem_entities = rem_data.groupby(self.entity_column).first()

            merged_entities = pd.concat([merged_entities, rem_entities])

        # concatenate single entities with merged entities
        merged_data = pd.concat([merged_entities, self.single_entities])

        # check if the final result contain a number of entities equal to the original dataset
        if len(merged_data[self.entity_column].unique()) != len(self.data[self.entity_column].unique()):
            raise Exception(
                "The data fusion task has generated a differnt number of entities with respect the original one.")

        return IntegratedDataset(merged_data, self.id_col, self.source_column, self.entity_column)

    def select_random_records(self, seed):
        """
        This method implements a data fusion technique by randomly selecting a single record among the ones that refer
        to the same entity.

        :param seed: the seed for the random choice
        :return: a new IntegratedDataset object where for each entity only one record is kept
        """

        pre_match_data = self.data.groupby(self.entity_column).filter(lambda x: len(x) > 1).index.values

        np.random.seed(seed)
        merged_data = self.data.groupby(self.entity_column).apply(lambda x: x.loc[np.random.choice(x.index), :])
        merged_data["match"] = merged_data.apply(lambda x: 1 if x["index"] in pre_match_data else 0, axis=1)

        return IntegratedDataset(merged_data, self.id_col, self.source_column, self.entity_column)

    def select_random_attribute_values(self, seed):
        """
        This method implements a data fusion technique by randomly selecting the attribute values of the records that
        refer to the same entity, in order to create a single output record per entity.

        :param seed: the seed for the random choices
        :return: a new IntegratedDataset object where for each entity only one record is kept
        """

        np.random.seed(seed)

        def get_record_by_random_attribute_values(x):

            template_row = x.iloc[0, :].copy()

            if len(template_row) > 1:

                all_columns = x.columns.values
                exclude_columns = ['index', 'id', 'source', 'entity_id']
                remaining_columns = list(set(all_columns).difference(set(exclude_columns)))

                for col in ['index', 'id', 'source']:
                    template_row[col] = None

                for col in remaining_columns:
                    record_id = np.random.choice(x.index.values)
                    record_col_val = x.loc[record_id, col]
                    template_row[col] = record_col_val

            return template_row

        merged_data = self.data.groupby(self.entity_column).apply(get_record_by_random_attribute_values)

        return IntegratedDataset(merged_data, self.id_col, self.source_column, self.entity_column)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    example = 'restaurant'
    EXAMPLE_DIR = os.path.join(DATA_DIR, example)

    # source 1
    file1 = os.path.join(EXAMPLE_DIR, "original", "zomato.csv")
    source1_id = 0

    # source 2
    file2 = os.path.join(EXAMPLE_DIR, "original", "yelp.csv")
    source2_id = 1

    # integrated data
    file3 = os.path.join(EXAMPLE_DIR, "original", "labeled_data.csv")

    random_seed = 24

    # prepare data
    data_prep_comp = DataPreparationComponent(file1, source1_id, file2, source2_id, file3)
    data_sources, integrated_data_sources = data_prep_comp.split_randomly_data(4, random_seed, mixed=False, debug=False)

    integration = integrated_data_sources[0]

    data_fusion_comp = DataFusionComponent(integration)
    # MERGE BY SOURCE
    merged_integration = data_fusion_comp.select_records_by_source_id(0)
    # MERGE BY RANDOM SELECTION
    # merged_integration = data_fusion_comp.select_random_records(random_seed)

    # check the results by selecting the top 5 entities wih multiple records in the original data and in the merged data
    entity_column = "entity_id"
    original_data = integration.get_data()
    original_entity_counts = original_data[entity_column].value_counts()
    original_entity_multiple_records_ids = original_entity_counts[original_entity_counts > 1].index.values[:5]
    original_entity_multiple_records = original_data[
        original_data[entity_column].isin(original_entity_multiple_records_ids)]
    print("UNMERGED DATA SAMPLE")
    print(original_entity_multiple_records)

    print("MERGED DATA SAMPLE")
    merged_integration_data = merged_integration.get_data()
    merged_entity_multiple_records = merged_integration_data[
        merged_integration_data[entity_column].isin(original_entity_multiple_records_ids)]
    print(merged_entity_multiple_records)
