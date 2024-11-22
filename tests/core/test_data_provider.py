import os

from mbmbm.core.dataset_provider import DatasetProvider


def test_data_provider(test_resources):
    print(os.environ["MBMBM_DATASET_DIR"])
    dp = DatasetProvider()
    configs = dp.get_configs("ds_test_small")
    assert "default" in configs
    assert "label_test" in configs
    print(configs)
    df_in, df_target = dp.get_dataframes(id="ds_test_small", stage="train")
    assert df_in.shape == (113, 764)
    assert df_target.shape == (113, 1)
    print(df_in)
    print(df_target)
