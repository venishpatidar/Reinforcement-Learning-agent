import os
import numpy as np
import matplotlib.pyplot as plt


def read_values_from_folder(folder_path):
    all_values = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith('train_avg'):  # Assuming files have a .txt extension
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                values = [float(line.strip()) for line in file]
                all_values.append(np.array(values))
    return all_values


if __name__=="__main__":
    # Data
    folder_path = './results/ApproximateQAgent'
    ApproximateQAgent_values = read_values_from_folder(folder_path)

    folder_path = './results/ReinforceAgent'
    ReinforceAgent_values = read_values_from_folder(folder_path)

    folder_path = './results/ActorCriticAgent'
    ActorCriticAgent_values = read_values_from_folder(folder_path)

    # Convert the lists into numpy arrays
    ApproximateQAgent_values = np.array(ApproximateQAgent_values)
    ReinforceAgent_values = np.array(ReinforceAgent_values)
    ActorCriticAgent_values = np.array(ActorCriticAgent_values)

    # Calculate mean and standard deviation across runs for t1, t2, and t3
    ApproximateQAgent_mean = np.mean(ApproximateQAgent_values, axis=0)
    ApproximateQAgent_std = np.std(ApproximateQAgent_values, axis=0)

    ReinforceAgent_mean = np.mean(ReinforceAgent_values, axis=0)
    ReinforceAgent_std = np.std(ReinforceAgent_values, axis=0)

    ActorCriticAgent_mean = np.mean(ActorCriticAgent_values, axis=0)
    ActorCriticAgent_std = np.std(ActorCriticAgent_values, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    # Mean
    plt.plot(ActorCriticAgent_mean, label='Actor Critic', marker='.')
    plt.plot(ReinforceAgent_mean, label='Reinforce', marker='.')
    plt.plot(ApproximateQAgent_mean, label='Approximate Q', marker='.')
    # Deviations
    plt.fill_between(range(len(ActorCriticAgent_mean)), ActorCriticAgent_mean - ActorCriticAgent_std, ActorCriticAgent_mean + ActorCriticAgent_std, alpha=0.2)
    plt.fill_between(range(len(ReinforceAgent_mean)), ReinforceAgent_mean - ReinforceAgent_std, ReinforceAgent_mean + ReinforceAgent_std, alpha=0.2)
    plt.fill_between(range(len(ApproximateQAgent_mean)), ApproximateQAgent_mean - ApproximateQAgent_std, ApproximateQAgent_mean + ApproximateQAgent_std, alpha=0.2)
    # Labels
    plt.xlabel('Episodes*100')
    plt.ylabel('Scores')
    # Misc
    plt.title('Approximate Q Agent, Reinforce Agent, and Actor Critic Agent')
    plt.legend()
    plt.grid(True)
    plt.show()
