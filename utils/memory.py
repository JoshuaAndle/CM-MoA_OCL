import math
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.distributed as dist
import numpy as np
from typing import Optional, Sized, Iterable, Tuple

class Memory:
    def __init__(self, args, data_source: Dataset=None) -> None:
        
        self.data_source = data_source
        if self.data_source is not None:
            self.images = []

        ### Moved memory size and # samples seen into the memory class to enable multiple buffers being tracked independently
        # self.memory_size = -1
        self.memory_size = args.get("memory_size", 0)
        self.memory_per_class = -1
        self.seen = {}

        self.memory = torch.empty(0)
        self.labels = torch.empty(0)
        self.cls_list = torch.empty(0)
        self.cls_count = {}
        self.cls_train_cnt = {}
        self.previous_idx = torch.empty(0)
        self.others_loss_decrease = torch.empty(0)


    ### Note: cls_list is not necessarily ordered as increasing or contiguous integers
    def add_new_class(self, cls_list: Iterable[int]) -> None:
        self.cls_list = torch.tensor(cls_list)
        # print("Resulting class list in memory: ", self.cls_list)
        for cls in cls_list:
            if cls not in self.cls_count.keys():
                self.cls_count[cls] = 0
            # if cls.item() not in self.cls_train_cnt.keys():
                self.cls_train_cnt[cls] = 0
                self.seen[cls] = 0


        # self.cls_count = torch.cat([self.cls_count, torch.zeros(len(self.cls_list) - len(self.cls_count))])
        # self.cls_train_cnt = torch.cat([self.cls_train_cnt, torch.zeros(len(self.cls_list) - len(self.cls_train_cnt))])

        _memory_per_class = self.memory_per_class
        self.memory_per_class = math.floor(self.memory_size/len(self.cls_list))

        if self.memory_per_class != _memory_per_class:
            print("New available memory per class: ", self.memory_per_class)
            ### Check if any previous classes need to have samples removed to allow new classes to be stored
            self.resize_class_memory()



    def remove_samples(self, cls, excess_count) -> None:
        class_indices = (self.labels == cls).nonzero(as_tuple=True)[0]
        random_removal_indices = torch.randperm(len(class_indices))
        ### Shuffle the indices corresponding to given class, then take the first ones to be removed
        print("Removing excess_count {} from cls: {}".format(excess_count, cls))
        random_removal_indices = class_indices[random_removal_indices][:excess_count]

        kept_indices = torch.arange(len(self.labels))
        kept_indices = kept_indices[~torch.isin(kept_indices, random_removal_indices)]
        self.memory = self.memory[kept_indices]
        self.labels = self.labels[kept_indices]
        self.cls_count[cls.item()] -= excess_count



    def resize_class_memory(self):
        for cls in self.cls_list:
            excess_count = self.cls_count[cls.item()] - self.memory_per_class
            if excess_count > 0:
                self.remove_samples(cls, excess_count)
        print("Class counts after resizing: ", self.cls_count)



    def replace_data(self, data: Tuple[Tensor, Tensor], idx: int=None) -> None:
        index, label = data
        if self.data_source is not None:
            image, label = self.data_source.__getitem__(index)
        if idx is None:
            if self.data_source is not None:
                self.images.append(image.unsqueeze(0))
            self.memory = torch.cat([self.memory, torch.tensor([index])])
            self.labels = torch.cat([self.labels, torch.tensor([label])])
            # self.cls_count[(self.cls_list == label).nonzero().squeeze()] += 1
            self.cls_count[label] += 1


            if self.cls_count[label] == 1:
                self.others_loss_decrease = torch.cat([self.others_loss_decrease, torch.tensor([0])])
            else:
                indice = (self.labels == label).nonzero().squeeze()
                self.others_loss_decrease = torch.cat([self.others_loss_decrease, torch.mean(self.others_loss_decrease[indice[:-1]]).unsqueeze(0)])
        else:
            if self.data_source is not None:
                self.images[idx] = image.unsqueeze(0)
            replaced_label = self.labels[idx]
            self.cls_count[replaced_label.item()] -= 1
            self.memory[idx] = index
            self.labels[idx] = label


            self.cls_count[label] += 1
            if self.cls_count[label] == 1:
                self.others_loss_decrease[idx] = torch.mean(self.others_loss_decrease)
            else:
                indice = (self.labels == label).nonzero().squeeze()
                self.others_loss_decrease[idx] = torch.mean(self.others_loss_decrease[indice[indice != idx]])





    def update_loss_history(self, loss: Tensor, prev_loss: Tensor, ema_ratio: float=0.90, dropped_idx: int=None) -> None:
        if dropped_idx is None:
            loss_diff = torch.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = torch.ones(len(loss), dtype=bool)
            mask[torch.tensor(dropped_idx, dtype=torch.int64).squeeze()] = False
            loss_diff = torch.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - torch.mean(self.others_loss_decrease[self.previous_idx.to(torch.int64)]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx.to(torch.int64)] -= (1 - ema_ratio) * difference
        self.previous_idx = torch.empty(0)
    
    def get_weight(self) -> Tensor:
        weight = torch.zeros(self.images.size(0))
        for cls in self.cls_list:
            weight[(self.labels == cls).nonzero().squeeze()] = 1 / (self.labels == cls).nonzero().numel()
        return weight

    def update_gss_score(self, score: int, idx: int=None) -> None:
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def __len__(self) -> int:
        return len(self.labels)

    def sample(self, memory_batchsize: int) -> Tuple[Tensor, Tensor]:
        assert self.data_source is not None
        idx = torch.randperm(len(self.images), dtype=torch.int64)[:memory_batchsize]
        images = []
        labels = []
        for i in idx:
            images.append(self.images[i])
            labels.append(self.labels[i])
        return torch.cat(images), torch.tensor(labels)

class DummyMemory(Memory):
    def __init__(self, data_source: Dataset=None, shape: Tuple[int, int, int]=(3, 32, 32), datasize: int=100) -> None:
        super(DummyMemory, self).__init__(data_source)
        self.shape = shape
        self.datasize = datasize
        self.images = torch.rand(self.datasize, *self.shape)
        self.labels = torch.randint(0, 10, (self.datasize,))
        self.cls_list = torch.unique(self.labels)
        self.cls_count = {}
        self.cls_train_cnt = {}
        self.others_loss_decrease = torch.zeros(self.datasize)


class MemoryBatchSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, batch_size: int, iterations: int = 1) -> None:
        self.memory = memory
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.randperm(len(self.memory), dtype=torch.int64)[:min(self.batch_size, len(self.memory))] for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] = int(self.memory.memory[idx])
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, samples_idx: int, batch_size: int, iterations: int = 1) -> None:
        self.samples_idx = samples_idx
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.randperm(len(self.samples_idx), dtype=torch.int64)[:min(self.batch_size, len(self.samples_idx))] for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] = int(self.samples_idx[idx])
    
    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

class MemoryOrderedSampler(torch.utils.data.Sampler):
    def __init__(self, memory: Memory, batch_size: int, iterations: int = 1) -> None:
        self.memory = memory
        self.batch_size = batch_size
        self.iterations = int(iterations)
        self.indices = torch.cat([torch.arange(len(self.memory), dtype=torch.int64) for _ in range(self.iterations)]).tolist()
        for i, idx in enumerate(self.indices):
            self.indices[i] =  int(self.memory.memory[idx])
    
    def __iter__(self) -> Iterable[int]:
        if dist.is_initialized():
            return iter(self.indices[dist.get_rank()::dist.get_world_size()])
        else:
            return iter(self.indices)
    def __len__(self) -> int:
        if dist.is_initialized():
            return len(self.indices[dist.get_rank()::dist.get_world_size()])
        else:
            return len(self.indices)