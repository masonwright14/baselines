import os
import tempfile

import zipfile
import tensorflow as tf
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load_for_multiple_nets_with_scope(path, scope):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act_params["scope"] = scope
        cur_graph = tf.Graph()
        with cur_graph.as_default():
            act = deepq.build_act(**act_params)
            sess = tf.Session(graph=cur_graph)
            with sess.as_default():
                with tempfile.TemporaryDirectory() as td:
                    arc_path = os.path.join(td, "packed.zip")
                    with open(arc_path, "wb") as f:
                        f.write(model_data)

                    zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
                    U.load_state(os.path.join(td, "model"))
        return ActWrapper(act, act_params), cur_graph, sess

    @staticmethod
    def load_for_multiple_nets(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        cur_graph = tf.Graph()
        with cur_graph.as_default():
            act = deepq.build_act(**act_params)
            sess = tf.Session(graph=cur_graph)
            with sess.as_default():
                with tempfile.TemporaryDirectory() as td:
                    arc_path = os.path.join(td, "packed.zip")
                    with open(arc_path, "wb") as f:
                        f.write(model_data)

                    zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
                    U.load_state(os.path.join(td, "model"))
        return ActWrapper(act, act_params), cur_graph, sess

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        cur_graph = tf.Graph()
        with cur_graph.as_default():
            act = deepq.build_act(**act_params)
            sess = tf.Session(graph=cur_graph)
            with sess.as_default():
                with tempfile.TemporaryDirectory() as td:
                    arc_path = os.path.join(td, "packed.zip")
                    with open(arc_path, "wb") as f:
                        f.write(model_data)

                    zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
                    U.load_state(os.path.join(td, "model"))
            return ActWrapper(act, act_params)

    @staticmethod
    def load_with_scope(path, scope):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act_params["scope"] = scope
        cur_graph = tf.Graph()
        with cur_graph.as_default():
            act = deepq.build_act(**act_params)
            sess = tf.Session(graph=cur_graph)
            sess.__enter__()
            with tempfile.TemporaryDirectory() as td:
                arc_path = os.path.join(td, "packed.zip")
                with open(arc_path, "wb") as f:
                    f.write(model_data)

                zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
                U.load_state(os.path.join(td, "model"))
            return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save_with_sess(self, sess, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            with sess.as_default():
                # print("Saving state now")
                # for var in tf.trainable_variables():
                #     print('normal variable: ' + var.op.name)
                U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

def load_for_multiple_nets_with_scope(path, scope):
    return ActWrapper.load_for_multiple_nets_with_scope(path, scope)

def load_for_multiple_nets(path):
    return ActWrapper.load_for_multiple_nets(path)

def load_with_scope(path, scope):
    return ActWrapper.load_with_scope(path, scope)

def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path)


def learn(env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          ep_mean_length=100,
          scope="deepq_train"):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = tf.Session()
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        scope=scope
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    with tempfile.TemporaryDirectory() as td:
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            with sess.as_default():
                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                with sess.as_default():
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                with sess.as_default():
                    update_target()

            mean_ep_reward = round(np.mean(episode_rewards[(-ep_mean_length - 1):-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean " + str(ep_mean_length) + " episode reward", mean_ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > ep_mean_length and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_ep_reward))
                    with sess.as_default():
                        U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            with sess.as_default():
                U.load_state(model_file)

    return act

def learn_multiple_nets(env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          ep_mean_length=100):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    cur_graph = tf.Graph()
    with cur_graph.as_default():
        sess = tf.Session(graph=cur_graph)
        sess.__enter__()

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph
        observation_space_shape = env.observation_space.shape
        def make_obs_ph(name):
            return U.BatchInput(observation_space_shape, name=name)

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.action_space.n,
        }

        act = ActWrapper(act, act_params)

        # Create the replay buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = max_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
            beta_schedule = None
        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        saved_mean_reward = None
        obs = env.reset()
        reset = True
        with tempfile.TemporaryDirectory() as td:
            model_saved = False
            model_file = os.path.join(td, "model")
            for t in range(max_timesteps):
                if callback is not None:
                    if callback(locals(), globals()):
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not param_noise:
                    update_eps = exploration.value(t)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, _ = env.step(env_action)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                if t > learning_starts and t % train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if prioritized_replay:
                        experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                    if prioritized_replay:
                        new_priorities = np.abs(td_errors) + prioritized_replay_eps
                        replay_buffer.update_priorities(batch_idxes, new_priorities)

                if t > learning_starts and t % target_network_update_freq == 0:
                    # Update target network periodically.
                    update_target()

                mean_ep_reward = round(np.mean(episode_rewards[(-ep_mean_length - 1):-1]), 1)
                num_episodes = len(episode_rewards)
                if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean " + str(ep_mean_length) + " episode reward", mean_ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

                if (checkpoint_freq is not None and t > learning_starts and
                        num_episodes > ep_mean_length and t % checkpoint_freq == 0):
                    if saved_mean_reward is None or mean_ep_reward > saved_mean_reward:
                        if print_freq is not None:
                            logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                       saved_mean_reward, mean_ep_reward))
                        U.save_state(model_file)
                        model_saved = True
                        saved_mean_reward = mean_ep_reward
            if model_saved:
                if print_freq is not None:
                    logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
                U.load_state(model_file)

        return act

def learn_and_save(env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          ep_mean_length=100,
          scope="deepq_train",
          path_for_save=None):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = tf.Session()
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        scope=scope
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    # for var in tf.trainable_variables():
    #     print('normal variable: ' + var.op.name)

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            with sess.as_default():
                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                # print("Reset env")
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # print("Minimize error")
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                with sess.as_default():
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                with sess.as_default():
                    # print("update target")
                    update_target()

            mean_ep_reward = round(np.mean(episode_rewards[(-ep_mean_length - 1):-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean " + str(ep_mean_length) + " episode reward", mean_ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            # print("Checkpoint")
            # for var in tf.trainable_variables():
            #     print('normal variable: ' + var.op.name)

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > ep_mean_length and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_ep_reward))
                    with sess.as_default():
                        # print("Saving current state")
                        U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            with sess.as_default():
                # print("Loading old state")
                U.load_state(model_file)

    # for var in tf.global_variables():
    #     print('all variables: ' + var.op.name)
    # for var in tf.trainable_variables():
    #     print('normal variable: ' + var.op.name)

    if path_for_save is not None:
        act.save_with_sess(sess, path=path_for_save)
    return act

def print_debug_info(scope_old, scope_new, sess):
    vars_in_scope_old = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_old)
    print("Variables in scope: " + scope_old)
    for v in vars_in_scope_old:
        print(v)

    vars_in_scope_new = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_new)
    print("\nVariables in scope: " + scope_new)
    for v in vars_in_scope_new:
        print(v)

    trainables = sess.run(tf.trainable_variables())
    print("\nTrainable variables:")
    for v in trainables:
        print(v)

def print_old_and_new_weights(scope_old, scope_new, sess):
    old_weights_name = scope_old + "/q_func/fully_connected/weights:0"
    old_weights = [v for v in tf.global_variables() if v.name == old_weights_name][0]
    print("old weights:")
    print(old_weights)
    old_value = sess.run(old_weights)
    print(old_value)

    new_weights_name = scope_new + "/q_func/fully_connected/weights:0"
    new_weights = [v for v in tf.global_variables() if v.name == new_weights_name][0]
    print("new weights:")
    print(new_weights)
    new_value = sess.run(new_weights)
    print(new_value)

    # copy_into_new_weights = new_weights.assign(old_weights)
    # sess.run(copy_into_new_weights)

    # new_value_updated = sess.run(new_weights)
    # print("updated value of new weights:")
    # print(new_value_updated)

def overwrite_new_net_with_old(scope_old, scope_new, sess):
    net_name_suffixes = [
        "eps:0",
        "/q_func/fully_connected/weights:0",
        "/q_func/fully_connected/biases:0",
        "/q_func/fully_connected_1/weights:0",
        "/q_func/fully_connected_1/biases:0",
        "/q_func/fully_connected_2/weights:0",
        "/q_func/fully_connected_2/biases:0"
    ]

    for suffix in net_name_suffixes:
        old_var_name = scope_old + suffix
        new_var_name = scope_new + suffix
        old_var = [v for v in tf.global_variables() if v.name == old_var_name][0]
        new_var = [v for v in tf.global_variables() if v.name == new_var_name][0]
        copy_var = new_var.assign(old_var)
        sess.run(copy_var)

        old_var_value = sess.run(old_var)
        updated_new_var_value = sess.run(new_var)
        if old_var_value != updated_new_var_value:
            raise ValueError("Not equal: " + str(old_var_value) + "\n" + \
                str(updated_new_var_value))

def retrain_and_save(env,
                     q_func,
                     lr=5e-4,
                     max_timesteps=100000,
                     buffer_size=50000,
                     exploration_fraction=0.5,
                     exploration_initial_eps=0.3,
                     exploration_final_eps=0.03,
                     train_freq=1,
                     batch_size=32,
                     print_freq=100,
                     checkpoint_freq=10000,
                     learning_starts=1000,
                     gamma=1.0,
                     target_network_update_freq=500,
                     prioritized_replay=False,
                     prioritized_replay_alpha=0.6,
                     prioritized_replay_beta0=0.4,
                     prioritized_replay_beta_iters=None,
                     prioritized_replay_eps=1e-6,
                     param_noise=False,
                     callback=None,
                     ep_mean_length=100,
                     scope_old="deepq_train",
                     scope_new="deepq_train_retrained",
                     prefix_for_save=None,
                     save_count=4):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    if save_count < 1:
        raise ValueError("save_count must be positive")

    sess = tf.Session()
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    act, train, update_target, _ = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        scope=scope_new
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from exploration_initial_eps.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=exploration_initial_eps,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()

    # overwrite q_func values
    # with values from the inner "q_func_old" of old network,
    # where q_func is a lambda around a network with fully_connected() and relu() parts.
    # q_func_old has scope of scope_old, and q_func has scope of scope_new.

    # print_debug_info(scope_old, scope_new, sess)
    overwrite_new_net_with_old(scope_old, scope_new, sess)
    # print_old_and_new_weights(scope_old, scope_new, sess)

    update_target()

    # for var in tf.trainable_variables():
    #     print('normal variable: ' + var.op.name)

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    save_iter = 1
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between
                # perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with
                # eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration,
                # Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t)
                    + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            with sess.as_default():
                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                # print("Reset env")
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # print("Minimize error")
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                with sess.as_default():
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                with sess.as_default():
                    # print("update target")
                    update_target()

            mean_ep_reward = round(np.mean(episode_rewards[(-ep_mean_length - 1):-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean " + str(ep_mean_length) + " episode reward", mean_ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            # print("Checkpoint")
            # for var in tf.trainable_variables():
            #     print('normal variable: ' + var.op.name)

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > ep_mean_length and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_ep_reward))
                    with sess.as_default():
                        # print("Saving current state")
                        U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_ep_reward
            if t > save_iter * (max_timesteps / save_count):
                cur_save_path = prefix_for_save + "_r" + str(save_iter) + ".pkl"
                act.save_with_sess(sess, path=cur_save_path)
                save_iter += 1
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            with sess.as_default():
                # print("Loading old state")
                U.load_state(model_file)


    # for var in tf.global_variables():
    #     print('all variables: ' + var.op.name)
    # for var in tf.trainable_variables():
    #     print('normal variable: ' + var.op.name)

    if prefix_for_save is not None:
        cur_save_path = prefix_for_save + "_r" + str(save_count) + ".pkl"
        act.save_with_sess(sess, path=cur_save_path)
    return act

def learn_retrain_and_save(env,
          q_func,
          lr=5e-4,
          max_timesteps_init=100000,
          buffer_size=50000,
          exploration_fraction=0.5,
          exploration_final_eps=0.03,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          ep_mean_length=100,
          scope="deepq_train",
          path_for_save=None,
          retrain_exploration_initial_eps=0.3,
          retrain_exploration_final_eps=0.03,
          retrain_save_count=4,
          max_timesteps_retrain=100000,
          retrain_config_str=None,
          prefix_for_save=None):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = tf.Session()
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    act, train, update_target, _ = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        scope=scope
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps_init + max_timesteps_retrain
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps_init),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    # for var in tf.trainable_variables():
    #     print('normal variable: ' + var.op.name)

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    with tempfile.TemporaryDirectory() as td:
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps_init + max_timesteps_retrain):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            phase_t = t % max_timesteps_init
            if not param_noise:
                update_eps = exploration.value(phase_t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(phase_t) + exploration.value(phase_t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            with sess.as_default():
                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                # print("Reset env")
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # print("Minimize error")
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                with sess.as_default():
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                with sess.as_default():
                    # print("update target")
                    update_target()

            mean_ep_reward = round(np.mean(episode_rewards[(-ep_mean_length - 1):-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean " + str(ep_mean_length) + " episode reward", mean_ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            # print("Checkpoint")
            # for var in tf.trainable_variables():
            #     print('normal variable: ' + var.op.name)

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > ep_mean_length and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_ep_reward))
                    with sess.as_default():
                        # print("Saving current state")
                        U.save_state(model_file)
                    saved_mean_reward = mean_ep_reward
            if t == max_timesteps_init:
                # change exploration curve
                exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps_retrain),
                                 initial_p=retrain_exploration_initial_eps,
                                 final_p=retrain_exploration_final_eps)
                # change opponent
                env.update_for_retrain(retrain_config_str)
                # reset state as if new game
                obs = env.reset()
                reset = True
                # save current network
                print("saving: " + path_for_save)
                act.save_with_sess(sess, path=path_for_save)
            elif t > max_timesteps_init and (t - max_timesteps_init) % (max_timesteps_retrain // retrain_save_count) == 0:
                # save current network under new name
                save_iter = (t - max_timesteps_init) // (max_timesteps_retrain // retrain_save_count)
                cur_save_path = prefix_for_save + "_r" + str(save_iter) + ".pkl"
                print("saving from retrain: " + cur_save_path)
                act.save_with_sess(sess, path=cur_save_path)
    return act
