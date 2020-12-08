import pytest

@pytest.fixture()
def mnist_dataset():
    import torchvision.datasets, torchvision.transforms
    from librl.utils import load_split_data
    class helper:
        def __init__(self):
            self.hypers = {}
            self.hypers['device'] = 'cpu'
            self.hypers['epochs'] = 1
            self.hypers['task_count'] = 1
            transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.1307, 0.3081)]) 
            # Construct dataloaders from datasets
            train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
            validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)
            self.t_loaders, self.v_loaders = load_split_data(train_dset, 100, 3), load_split_data(validation_dset, 1000, 1)
    
    return helper()