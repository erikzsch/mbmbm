from hydra.utils import instantiate
from omegaconf import DictConfig

from mbmbm import logger
from mbmbm.featureengine.featureenginebase import FeatureEngineBase
from mbmbm.featureengine.featureengines import FeatureSelectorFromModel


def init_feature_engine(cfg: DictConfig, model):
    """
    Initialize the feature engineering pipeline based on the configuration provided.

    :param cfg: Configuration object containing details about the feature engineering pipeline.
    :type cfg: DictConfig
    :param model: The machine learning model used to instantiate the feature engine if required.
    :type model: object
    :return: An instance of a FeatureEngineBase derived object.
    :rtype: FeatureEngineBase

    :raises AssertionError: If the provided feature_engine configuration is a string and not equal to 'from_model'.
    """
    if isinstance(cfg.feature_engine, str):
        # case: cfg.feature_engine: from_model
        assert cfg.feature_engine == "from_model", "If cfg.feature_engine is a string, only 'from_model' is allowed"
        logger.info("Instantiating feature engine from model.")
        try:
            feature_engine = FeatureSelectorFromModel(model=model)
            logger.success("Successfully instantiated FeatureSelectorFromModel feature engine.")
        except Exception as e:
            logger.error(f"Error occurred while instantiating FeatureSelectorFromModel: {e}")
            raise
    else:
        # case: cfg.feature_engine:
        #          _target_: ...
        logger.info(f"Instantiating feature engine from cfg: {cfg.feature_engine}")
        try:
            feature_engine: FeatureEngineBase = instantiate(cfg.feature_engine)
            logger.success("Successfully instantiated feature engine from configuration.")
        except Exception as e:
            logger.error(f"Error occurred while instantiating feature engine from configuration: {e}")
            raise

    logger.debug(f"Feature engine instance: {feature_engine}")
    return feature_engine
