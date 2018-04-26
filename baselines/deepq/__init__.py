from baselines.deepq import models  # noqa
from baselines.deepq.build_graph import build_act, build_train  # noqa
# from baselines.deepq.simple import learn, load  # noqa
from baselines.deepq.simple_conservative import learn, load, load_for_multiple_nets, \
    learn_multiple_nets, learn_and_save, load_with_scope, load_for_multiple_nets_with_scope
    # noqa
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
