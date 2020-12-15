import argparse
import enum
import functools
import os

import dill
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
import imgfiltrl.nn

# Log replay information to harddisk
# This will enable us to generate graphs in the future.
class DirLogger:
    def __init__(self, log_dir):
        pass
        self.log_dir = log_dir
        os.makedirs(self.log_dir)

    def __call__(self, epochs, task_samples):
        subdir = os.path.join(self.log_dir, f"epoch-{epochs}")
        # I would be very concerned if the subdir already exists
        os.makedirs(subdir)
        for task_idx, task in enumerate(task_samples):
            task_subdir = os.path.join(subdir, f"task-{task_idx}")
            os.makedirs(task_subdir)
            for trajectory_idx, trajectory in enumerate(task.trajectories):
                with open(os.path.join(task_subdir, f"traj{trajectory_idx}.pkl"), "wb") as fptr:
                    dill.dump(trajectory, fptr)


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

######################
# Neural Net Helpers #
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

# Create the inner classifier network.
def gen_classifier(dims, labels):
    layers = [
        librl.nn.core.cnn.conv_def(4, 4, 1, 2, 1, False),
        librl.nn.core.cnn.conv_def(4, 4, 1, 2, 1, False),
        librl.nn.core.cnn.pool_def(2, 2, 0, 2, True, 'max'),
        librl.nn.core.cnn.conv_def(8, 8, 2, 4, 1, False),
        librl.nn.core.cnn.conv_def(8, 8, 2, 4, 1, False),
        librl.nn.core.cnn.pool_def(2, 2, 0, 1, True, 'max'),
    ]

    main_cnn = librl.nn.core.ConvolutionalKernel(layers, dims[1:], dims[0])
    aug_cnn = librl.nn.core.ConvolutionalKernel(layers, dims[1:], 1)
    stacker = imgfiltrl.nn.FlattenCatKernel(main_cnn, aug_cnn)
    class_mlp = librl.nn.core.MLPKernel(functools.reduce(lambda x,y: x*y, stacker.output_dimensions, 1), [32], dropout=.2)
    class_kernel = librl.nn.core.sequential.SequentialKernel([stacker, class_mlp])
    return librl.nn.classifier.Classifier(class_kernel, labels)

#######################
#     Entry point     #
#######################
def main(args):
    hypers = {}
    hypers['device'] = 'cpu'
    hypers['epochs'] = args.epochs
    # 
    hypers['task_count'] = 1
    # Number of actions that will be suggested each epoch
    hypers['episode_length'] = args.episode_length
    # Number of epochs classifiers will be trained for each timestep
    hypers['adapt_steps'] = args.adapt_steps
    dset = mnist_dataset()
    labels, dims = 10,dset.dims
    network0, network1 = gen_classifier(dims, labels), gen_classifier(dims, labels)
    env = imgfiltrl.env.ImageClassifictionEnv(network0, network1, 
        torch.nn.CrossEntropyLoss(), train_dataset=dset.train_dset, 
        validation_dataset=dset.validation_dset, normalize_fn=dset.normalize_fn,
        adapt_steps=args.adapt_steps
    )

    critic_kernel= librl.nn.core.MLPKernel(4*3, [100, 100])
    critic = librl.nn.critic.ValueCritic(critic_kernel)
    actor_kernel= librl.nn.core.RecurrentKernel(4*3, 100, 3)
    actor = imgfiltrl.actor.FilterTreeActor(actor_kernel, env.observation_space)
    agent = args.alg(hypers, critic, actor)
    agent.train()

    # Create an environment in which to train our agent.
    # This environment is what will apply filters and train our classifier
    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=args.episode_length, 
        replay_ctor=imgfiltrl.replay.ProductEpisodeWithExtraLogs))
    # Kick off training.
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step, log_fn=DirLogger(args.log_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train agent to apply effective filters to MNIST photos for image classification.")
    parser.add_argument("--log-dir", dest="log_dir", help="Directory in which to store logs. Must not already exist")
    # Choose RL algorthim.
    learn_alg_group = parser.add_mutually_exclusive_group()
    learn_alg_group.add_argument("--vpg", action='store_const', const=vpg_helper, dest='alg', help="Train agent using VPG.")
    learn_alg_group.add_argument("--pgb", action='store_const', const=pgb_helper, dest='alg', help="Train agent using PGB.")
    learn_alg_group.set_defaults(alg=vpg_helper)
    # Task distribution hyperparams.
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for which to train the filter generating network.")
    parser.add_argument("--episode-length", dest="episode_length", default=2, type=int, help="Number of timestps per episode.")
    parser.add_argument("--adapt-steps", dest="adapt_steps", default=2, type=int, help="Number of epochs for which to the classifier networks.")
    args = parser.parse_args()
    main(args)
