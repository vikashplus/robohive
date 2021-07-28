import gym
import mj_envs.envs.biomechanics
import pickle
import pytest

GYM_ENVIRONMENT_IDS = (
    'FingerReachMotorFixed-v0',
    'FingerReachMotorRandom-v0',
    'FingerReachMuscleFixed-v0',
    'FingerReachMuscleRandom-v0',
    'FingerPoseMotorFixed-v0',
    'FingerPoseMotorRandom-v0',
    'FingerPoseMuscleFixed-v0',
    'FingerPoseMuscleRandom-v0',
    'IFTHPoseMuscleRandom-v0',
    'HandPoseAMuscleFixed-v0',
    'IFTHKeyTurnFixed-v0',
    'IFTHKeyTurnRandom-v0',
    'HandObjHoldFixed-v0',
    'HandObjHoldRandom-v0',
    'HandPenTwirlFixed-v0',
    'HandPenTwirlRandom-v0',
    'BaodingFixed-v1',
    'BaodingFixed4th-v1',
    'BaodingFixed8th-v1',
)


@pytest.mark.parametrize("environment_id", GYM_ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env1 = gym.make(environment_id)
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.action_space == env2.action_space, (
        env1.action_space, env2.action_space)

