import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CosmosSDK-v0',
    entry_point='cosmos_sdk_gym.envs:CosmosSDKEnv'
)
