import pytest
import torch.nn

import librl.nn.core, librl.nn.classifier
import librl.task
import librl.train.train_loop, librl.train.classification

from . import *

###################
# GPU Based Tests #
###################
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA.")
def test_label_images(mnist_dataset):
    hypers = mnist_dataset.hypers
    hypers['device'] = 'cuda'
    class_kernel = librl.nn.core.MLPKernel((1,28,28), (200, 100))
    class_net = librl.nn.classifier.Classifier(class_kernel, 10)
    class_net = class_net.to("cuda")

    t,v = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    
    # Construct a labelling task.
    dist = librl.task.distribution.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ClassificationTask, 
        classifier=class_net, criterion=torch.nn.CrossEntropyLoss(),
        train_data_iter=t, validation_data_iter=v,train_percent=.2, 
        validation_percent=.1, device="cuda")
    )
    librl.train.train_loop.cls_trainer(hypers, dist, librl.train.classification.train_single_label_classifier)
