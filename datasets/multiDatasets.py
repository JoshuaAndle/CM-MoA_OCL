# Multi datasets for continual learning
# All datasets needs to be in the same format.
# have targets and classes within the dataset.

from typing import Callable, Optional, Iterable
from torch.utils.data import Dataset


class multiDatasets(Dataset):

    def __init__(
        self,
        datasets: Iterable[Dataset],
        root: Optional[str] = None,
        train: Optional[bool] = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: Optional[bool] = False,
        preprocessed: bool = False,
    ) -> None:

        super().__init__()
        self.datasets = []
        self.dataset_lengths = []
        self.classes = []

        class_count = 0
        self.class_count_by_dataset = []
        if preprocessed:
            print("Setting up multiDataset with preprocessed datasets.")
            for dataset in datasets:
                if not isinstance(dataset, Dataset):
                    raise TypeError("dataset should be a Dataset object")
                print("Dataset number of classes: ", len(dataset.classes_names))
                print("Dataset type of targets: ", type(dataset.targets))
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(self.datasets[-1]))
                class_count += len(self.datasets[-1].classes_names)
                self.class_count_by_dataset.append(len(self.datasets[-1].classes_names))

        else:
            for dataset in datasets:
                if not isinstance(dataset, Dataset):
                    raise TypeError("dataset should be a Dataset object")
                self.datasets.append(dataset(root, train, transform, target_transform, download))
                self.dataset_lengths.append(len(self.datasets[-1]))
                class_count += len(self.datasets[-1].classes_names)

        #!# Going off of previous partially implemented code. Not sure why they had this as a string, but until I know where else its used I have added class_count_by_dataset to instead be used in incrementing target IDs
        self.classes = [str(i) for i in range(len(self.classes))]
        self.targets = []
        self.sample_task_ids = []
        self.classes_names = []

        for t, dataset in enumerate(self.datasets):
            self.classes_names += dataset.classes_names
            # for cls in dataset.targets:
                # self.targets.append(int(cls) + sum(self.classes[:i]))
                # self.sample_task_ids.append(i)
            ### Add all the targets for dataset t, offset by the amount of classes in preceding tasks' datasets
            for i in dataset.targets:
                self.targets.append(int(i) + sum(self.class_count_by_dataset[:t]))
                self.sample_task_ids.append(t)
                

    def __getitem__(self, index):
        target = self.targets[index]
        ### Set up this way rather than a single list/tensor as dataset image sizes may be incompatible
        for i, dataset in enumerate(self.datasets):
            if index < self.dataset_lengths[i]:
                ### Allow the dataset class to apply any transforms, then return the image and offset target label
                image, _ = dataset[index]
                return image, target
                # return dataset[index], target
            index -= self.dataset_lengths[i]
        print("Item not found in multidataset!")

    def __len__(self):
        return len(self.targets)
