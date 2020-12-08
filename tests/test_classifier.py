import librl.nn.core, librl.nn.classifier
import librl.reward
import librl.task
import librl.train.train_loop, librl.train.cc
import librl.train.classification
import torch, torch.utils


# Modified version of librl/tests/reco_test_label.
# Confirms that the image reco functionality of librl
# has been correctly configured inside nasrl.
def test_label_mnist(mnist_dataset):

    class_kernel = librl.nn.core.MLPKernel((1, 28, 28), (200, 100))
    class_net = librl.nn.classifier.Classifier(class_kernel, 10)

    dist = librl.task.TaskDistribution()
    # Construct dataloaders from datasets
    t,v = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    # Construct a labelling task.
    dist.add_task(librl.task.Task.Definition(librl.task.ClassificationTask, 
        classifier=class_net, criterion=torch.nn.CrossEntropyLoss(), 
        train_data_iter=t, validation_data_iter=v, train_percent=.1, 
        validation_percent=.1)
    )
    librl.train.train_loop.cls_trainer(mnist_dataset.hypers, dist, librl.train.classification.train_single_label_classifier)