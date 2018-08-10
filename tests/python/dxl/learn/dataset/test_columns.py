import pytest


@pytest.mark.skip('Not implemented yet.')
def test_convert_columns_to_dataset():
    """
        processing :: Columns c => Tensor | Tuple[Tensor] | Namedtuple[Tensor] | Dict[Tensor] 

        One may define some helper functions like:
        columns_to_tf_dataset :: Columns :: -> tf.Dataset (most case via tf.Dataset.from_generator)
        standard_tf_dataset_processing :: tf.Dataset -> tf.Dataset
        dataset2tensor :: tf.Dataset -> tf.Tensor
    """
    columns = TestColumns()
    dataset = dataset.Dataset.from_columns_with(columns, processing)
