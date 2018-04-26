import gym

from baselines import deepq

def callback(lcl, glb):
    # stop training if reward exceeds 0.5
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 0.5
    return is_solved

def main():
    env_name = "Connect4MaxConvAug-v0"
    # env_name = "Connect4MaxConvAugD2-v0"
    print("Environment: " + env_name)
    env = gym.make(env_name)
    model = deepq.models.cnn_to_mlp(
        # convs=[(32, 4, 1), (64, 4, 1), (64, 3, 1)],
        convs=[(32, 4, 1), (32, 4, 1)],
        hiddens=[256],
        dueling=True,
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-2,
        max_timesteps=15000,
        buffer_size=5000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        print_freq=10,
        param_noise=False,
        gamma=0.99,
        callback=callback
    )
    print("Saving model to c4_deepq_model.pkl")
    act.save("c4_deepq_model.pkl")

if __name__ == '__main__':
    main()
