import argparse
import enum
import functools
import pickle
import os

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import pytest
import torch, torch.utils

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.reward, librl.replay.episodic
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc
import librl.nn.classifier

import imgfiltrl.actor
import imgfiltrl.env
import imgfiltrl.replay

class DirLogger:
    def __init__(self, log_dir):
        pass
        #self.log_dir = log_dir
        #os.makedirs(self.log_dir)

    def __call__(self, epochs, task_samples):
        return
        subdir = os.path.join(self.log_dir, f"epoch-{epochs}")
        # I would be very concerned if the subdir already exists
        os.makedirs(subdir)
        accuracy_list = []
        for task_idx, task in enumerate(task_samples):
            task_subdir = os.path.join(subdir, f"task-{task_idx}")
            os.makedirs(task_subdir)
            for trajectory_idx, trajectory in enumerate(task.trajectories):
                with open(os.path.join(task_subdir, f"traj{trajectory_idx}.pkl"), "wb") as fptr:
                    pickle.dump(trajectory, fptr)
                    accuracy_list.append(trajectory.extra[len(trajectory.extra)-1]['accuracy'][-1])
        print(f"Average accuracy for epoch {epochs} was {sum(accuracy_list)/len(accuracy_list)}.")


######################
#      Datasets      #
######################
def mnist_dataset():
    import torchvision.datasets, torchvision.transforms
    from librl.utils import load_split_data
    class helper:
        def __init__(self):
            # Construct dataloaders from datasets
            self.train_dset = torchvision.datasets.MNIST("__pycache__/MNIST",
                transform=torchvision.transforms.transforms.ToTensor(), download=True
            )
            self.validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST",
                transform=torchvision.transforms.transforms.ToTensor(), download=False
            )
            self.normalize_fn = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(0.1307, 0.3081)]) 
            self.dims = (1, 28, 28)
    return helper()

def cifar10_dataset():
    import torchvision.datasets, torchvision.transforms
    from librl.utils import load_split_data
    class helper:
        def __init__(self):
            mean, stdev = .5, .25
            transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((mean, mean, mean), (stdev, stdev, stdev))]) 

            # Construct dataloaders from datasets
            train_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, download=True)
            validation_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, train=False)
            self.t_loaders, self.v_loaders = load_split_data(train_dset, 100, 3), load_split_data(validation_dset, 1000, 1)
            self.dims = (3, 32, 32)
    return helper()

######################
# Execute train loop #
######################
def vpg_helper(hypers, _, policy_net):
    agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent.train()
    return agent
    

def pgb_helper(hypers, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PGB(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

def ppo_helper(hypers, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PPO(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

def gen_classifier(dims, labels):
    layers = [
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.pool_def(1, 1, 0, 1, True, 'max'),
    ]

    class_cnn = librl.nn.core.ConvolutionalKernel(layers, dims[1:], dims[0])
    class_mlp = librl.nn.core.MLPKernel(functools.reduce(lambda x,y: x*y, class_cnn.output_dimension, 1))
    class_kernel = librl.nn.core.sequential.SequentialKernel([class_cnn, class_mlp])
    return librl.nn.classifier.Classifier(class_kernel, labels)

#######################
#     Entry point     #
#######################
def main(args):
    hypers = {}
    hypers['device'] = 'cpu'
    hypers['epochs'] = args.epochs
    hypers['task_count'] = args.task_count
    hypers['adapt_steps'] = hypers['episode_length'] = args.adapt_steps
    hypers['max_mlp_layers'] = 10
    hypers['max_cnn_layers'] = 10
    dset = mnist_dataset()
    labels, dims = 10,dset.dims
    network0, network1 = gen_classifier(dims, labels), gen_classifier(dims, labels)
    env = imgfiltrl.env.ImageClassifictionEnv(network0, network1, 
        torch.nn.CrossEntropyLoss(), train_dataset=dset.train_dset, 
        validation_dataset=dset.validation_dset, normalize_fn=dset.normalize_fn
    
    )

    layers = [
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.pool_def(1, 1, 0, 1, True, 'max'),
    ]

    critic_kernel= librl.nn.core.MLPKernel(4*3, [100, 100])
    critic = librl.nn.critic.ValueCritic(critic_kernel)
    actor_kernel= librl.nn.core.MLPKernel(4*3, [100, 100])
    actor = imgfiltrl.actor.FilterTreeActor(actor_kernel, env.observation_space)
    agent = args.alg(hypers, critic, actor)
    agent.train()

    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=args.adapt_steps, 
        replay_ctor=imgfiltrl.replay.ProductEpisodeWithExtraLogs))
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step, )#log_fn=DirLogger(args.log_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generator network on a task for multiple epochs and record results.")
    parser.add_argument("--log-dir", dest="log_dir", help="Directory in which to store logs.")
    # Choose RL algorthim.
    learn_alg_group = parser.add_mutually_exclusive_group()
    learn_alg_group.add_argument("--vpg", action='store_const', const=vpg_helper, dest='alg', help="Train agent using VPG.")
    learn_alg_group.add_argument("--pgb", action='store_const', const=pgb_helper, dest='alg', help="Train agent using PGB.")
    learn_alg_group.add_argument("--ppo", action='store_const', const=ppo_helper, dest='alg', help="Train agent using PPO.")
    learn_alg_group.set_defaults(alg=vpg_helper)
    # Task distribution hyperparams.
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for which to train the generator network.")
    parser.add_argument("--adapt-steps", dest="adapt_steps", default=20, type=int, help="Number of epochs which to the generated network.")
    parser.add_argument("--task-count", dest="task_count", default=3, type=int, help="Number of times of trials of the generator per epoch.")
    args = parser.parse_args()
    main(args)