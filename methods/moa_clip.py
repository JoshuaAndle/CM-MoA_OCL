import gc
import random
import time
import logging
import datetime
import os.path as osp
# from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.models as models

from methods._trainer import _Trainer
from utils.train_utils import select_optimizer, select_scheduler, exp_lr_scheduler
from utils.memory import Memory, MemoryBatchSampler
# from utils.memory import MemoryBatchSampler

from models.AutoEncoder import AutoEncoder, Alexnet_FE

logger = logging.getLogger()


#!# Used for both lora-clip and adapter-clip. 
###     The only difference is that peft_method is lora for lora_clip and adapter for adapter_clip
class MoACLIP(_Trainer):

    def __init__(self, **kwargs):
        super(MoACLIP, self).__init__(**kwargs)
        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.visible_classes = self.args.get('visible_classes', 'batch')

        ### Track which experts have been frozen and which have been assigned to each task
        # self.frozen_experts = {"text":{}, "visual":{}}
        self.experts_by_task = {}
        self.autoencoders = {}
        self.optimizer_encoder = None

        ### Pretrained Alexnet model used for producing features for AutoEncoder inputs
        pretrained_alexnet = models.alexnet(pretrained=True)
        for k, v in pretrained_alexnet.named_parameters():
            v.requires_grad = False
        # Derives a feature extractor model from the Alexnet model
        self.feature_extractor = Alexnet_FE(pretrained_alexnet)

        self.batch_counter = 0


        self.task_by_subnet = {}


    
    def select_subnet(self, task_id):
        for subnet, task_list in self.task_by_subnet.items():
            print("Checking subnet {} and task_list {}".format(subnet, task_list))
            if task_id in task_list:
        if len(self.task_by_subnet.keys()) == 0:
            subnet = 0
            self.task_by_subnet[0] = [task_id]
        else:
            subnet = max(self.task_by_subnet.keys()) + 1
            self.task_by_subnet[subnet] = [task_id]

        print("Selecting subnetwork: ", subnet, " for task ID: ", task_id)
        return subnet


    ### Prior to task, set task ID in model and initialize new expert mask, task routers, and task autoencoder
    def online_before_task(self, task_id):
        ### Deferred assignment since self.device isnt set up at init
        self.feature_extractor.to(self.device)
        subnet = self.select_subnet(task_id)

        ### Add a memory buffer for the new task and set it as the active buffer
        # self.memory_by_task[task_id] = Memory()
        # self.memory = self.memory_by_task[task_id]

        self.model.clip_model.set_subnet(subnet)
        self.autoencoders[task_id] = AutoEncoder()
        self.optimizer_encoder = optim.Adam(self.autoencoders[task_id].parameters(), lr=0.003, weight_decay=0.0001)

        self.experts_by_task[task_id] = {"text":{}, "visual":{}}
        for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
            self.experts_by_task[task_id]['text'][i] = []
            self.experts_by_task[task_id]['visual'][i] = []
            ### Set up new routers for new task in all transformer residual blocks
            self.model.clip_model.visual.transformer.resblocks[i].init_router()
            self.model.clip_model.transformer.resblocks[i].init_router()


        ### Produce a list of all experts frozen for tasks other than the current one
        frozen_experts = []
        for key in self.experts_by_task.keys():
            ### This prevents revisited tasks from training experts frozen in later tasks, alternatively may want to use "key < task_id"
            if key != task_id:
                for modal in ["text", "visual"]:
                    for block in self.experts_by_task[key][modal].keys():
                        frozen_experts.extend(self.experts_by_task[key][modal][block])
        print("The number of frozen experts is: ", len(frozen_experts))
        # print("The list of frozen experts is: ", frozen_experts)





        # Freeze some parameters
        for k, v in self.model.named_parameters():
            if k in frozen_experts:
                v.requires_grad = False
                continue
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False

        logger.info("Total parameters:\t{}".format(sum(p.numel() for p in self.model.parameters())))
        logger.info("Trainable parameters:\t{}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.reset_opt()


    def online_after_task(self, task_id):
        
        print("Memory buffer length after training: ", self.memory.labels.shape)
        print("Memory buffer labels: ", self.memory.labels.unique())
        sorted_counts = dict(sorted(self.memory.cls_count.items(), key=lambda item: item[1], reverse=True))
        print("Memory class counts: ", sorted_counts)


        ### Assign the most-used experts for the current task for freezing
        for i in range(len(self.model.clip_model.visual.transformer.resblocks)):
            visual_choose_map = self.model.clip_model.visual.transformer.resblocks[i].choose_map_image
            text_choose_map = self.model.clip_model.transformer.resblocks[i].choose_map_text
            top_values_v, top_indices_v = torch.topk(visual_choose_map, 2)
            top_values_t, top_indices_t = torch.topk(text_choose_map, 2)

            for j in range(len(top_indices_v)):
                self.experts_by_task[task_id]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i,top_indices_v[j]))
                self.experts_by_task[task_id]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i,top_indices_v[j]))
                self.experts_by_task[task_id]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i,top_indices_v[j]))
                self.experts_by_task[task_id]["visual"][i].append('visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i,top_indices_v[j]))
            for k in range(len(top_indices_t)):
                self.experts_by_task[task_id]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i, top_indices_t[k]))
                self.experts_by_task[task_id]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i, top_indices_t[k]))
                self.experts_by_task[task_id]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i, top_indices_t[k]))
                self.experts_by_task[task_id]["text"][i].append('transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i, top_indices_t[k]))








    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        self.model.update_class_names(self.exposed_classes_names)

        # self.memory_sampler = MemoryBatchSampler(self.memory, self.memory_batchsize, self.temp_batchsize * self.online_iter * self.world_size)
        # self.memory_dataloader = DataLoader(self.train_dataset,
        #                                     batch_size=self.memory_batchsize,
        #                                     sampler=self.memory_sampler,
        #                                     num_workers=4)
        # self.memory_provider = iter(self.memory_dataloader)

        self.batch_counter += 1



        self.memory_sampler = MemoryBatchSampler(self.memory, self.memory_batchsize,
                                    self.temp_batchsize * self.online_iter * self.world_size)
        self.memory_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.memory_batchsize,
                                            sampler=self.memory_sampler,
                                            num_workers=4)
        self.memory_provider = iter(self.memory_dataloader)


        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        self.update_memory(idx, labels)
        del (images, labels)
        gc.collect()
        # torch.cuda.empty_cache()
        return _loss / _iter, _acc / _iter



    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        if self.visible_classes == 'batch':
            # batch
            train_class_list = self.batch_exposed_classes
            train_class_name_list = self.batch_exposed_classes_names

        else:
            # all
            train_class_list = self.exposed_classes
            train_class_name_list = self.exposed_classes_names

        x, y = data


        ### My understanding is this is making a contiguous set of class labels using the indices of train_class_list
        for j in range(len(y)):
            y[j] = train_class_list.index(y[j].item())

        x,y = x.to(self.device), y.to(self.device)
        x = self.train_transform(x)




        text_tokens = self.model.labels_tokenize(train_class_name_list)

        current_autoencoder = self.autoencoders[self.task_id]


        ### Train autoencoder only on the current task data for later task re-identification 
        input_to_ae = self.feature_extractor(x)
        input_to_ae = F.sigmoid(input_to_ae.view(input_to_ae.size(0), -1).to(self.device))

        self.optimizer_encoder = exp_lr_scheduler(self.optimizer_encoder, self.batch_counter, 0.01)
        self.optimizer_encoder.zero_grad()
        current_autoencoder.zero_grad()

        current_autoencoder.to(self.device)
        outputs = current_autoencoder(input_to_ae)

        encoder_criterion = nn.MSELoss()
        loss_autoencoder = encoder_criterion(outputs, input_to_ae)
        loss_autoencoder.backward()
        self.optimizer_encoder.step()







        ### Train clip model adapters
        # with torch.cuda.amp.autocast(enabled=self.use_amp):
        with torch.amp.autocast('cuda', enabled=self.use_amp):            
            logit, image_features, text_features = self.model(x, text_tokens)
            loss = self.criterion(logit, y)
        _, preds = logit.topk(self.topk, 1, True, True)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        ### Only update scheduler once the memory batch has been run
        # self.update_schedule()



        if self.args.get('grad_analysis', False):
            self._grad_analysis(image_features.clone().detach(),
                                text_features.clone().detach(),
                                y.clone().detach(), train_class_list)

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)




        ### If there is a memory buffer, we also do a batch of memory samples
        ### We do this in two passes rather than concatenate batches since datasets may have incompatible sizes of images


        if len(self.memory) > 0 and self.memory_batchsize > 0:
            x_mem, y_mem = next(self.memory_provider)
            for i in y_mem.unique():
                if i not in train_class_list:
                    train_class_list.append(i)
                    train_class_name_list.append(self.exposed_classes_names[self.exposed_classes.index(i)])


            ### My understanding is this is making a contiguous set of class labels using the indices of train_class_list
            for j in range(len(y_mem)):
                y_mem[j] = train_class_list.index(y_mem[j].item())

            x_mem, y_mem = x_mem.to(self.device), y_mem.to(self.device)
            x_mem = self.train_transform(x_mem)



            ### Update tokens to reflect potential newly added classes from memory buffer
            text_tokens = self.model.labels_tokenize(train_class_name_list)


            

            ### Train clip model adapters
            # with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.amp.autocast('cuda', enabled=self.use_amp):            
                logit, image_features, text_features = self.model(x_mem, text_tokens)
                loss = self.criterion(logit, y_mem)
            _, preds = logit.topk(self.topk, 1, True, True)

            ### Accumulate gradients with the online batch and memory batch before zeroing again
            # self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            total_loss += loss.item()
            total_correct += torch.sum(preds == y_mem.unsqueeze(1)).item()
            total_num_data += y_mem.size(0)


        self.update_schedule()


        return total_loss, total_correct / total_num_data






    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        pred_list = []

        # offset = self.label_offset[self.task_id] if self.per_task_datasets else 0

        self.model.eval()
        current_sample_count = 0
        with torch.no_grad():
            start_time = time.time()
            for i, data in enumerate(test_loader):
                if self.debug and i >= 5:
                    break
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)
                current_sample_count += len(y)

                logit, _, _ = self.model(x, is_train=False, val_task_id=self.task_id)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)
                
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                    
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()
                pred_list += pred.tolist()
                print("Batch time after ", current_sample_count, " samples: ", time.time()-start_time,flush=True)
                start_time = time.time()


        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        cm = confusion_matrix(label, pred_list)

        eval_dict = {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "cls_acc": cls_acc,
            "confusion_matrix": cm.tolist()
        }
        
        print("Processing time: ", time.time()-start_time,flush=True)

        return eval_dict


    def offline_evaluate(self, test_loader, classes_names):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []

        text_tokens = self.model.labels_tokenize(classes_names)
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)


                # predict batch image domain:
                input_to_ae = self.feature_extractor(y)
                input_to_ae = F.sigmoid(input_to_ae.view(input_to_ae.size(0), -1))

                encoder_criterion = nn.MSELoss()
                
                model_autoencoder = self.autoencoders[0]
                outputs = model_autoencoder(input_to_ae)
                best_l = encoder_criterion(outputs, input_to_ae)
                best_router = 0
                for i in range(1, 12):
                    outputs = self.autoencoders[i](input_to_ae)
                    new_l = encoder_criterion(outputs, input_to_ae)
                    if new_l < best_l:
                        best_l = new_l
                        best_router = i

                ### Changed to defer checking if zero-shot CLIP should be used until after all known tasks are checked
                if best_l > self.args.threshold:
                    best_router = -1

                task_id = best_router
                self.model.clip_model.set_task(task_id)

                ### Evaluate using model set to the identified best-fit known task routing
                logit, _, _ = self.model(x, text_tokens, is_train=False, val_task_id=task_id)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()
                pred_list += pred.tolist()

        total_acc = total_correct / total_num_data

        return total_acc







    ### Updates memory, just tracking indices and labels
    def update_memory(self, sample, label):
        # print("Updating memory")
        # Update memory
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.memory.seen[lbl.item()] += 1
                # if len(self.memory) < self.memory_size:
                if self.memory.cls_count[lbl.item()] < self.memory.memory_per_class:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.memory.seen[lbl.item()], (1, )).item()
                    ### Get the jth index used by that class to maintain class balance in the buffer
                    if j < self.memory.memory_per_class:
                        valid_cls_idxs = (self.memory.labels == lbl).nonzero(as_tuple=True)[0]
                        idx.append(valid_cls_idxs[j])
                    else:
                        idx.append(self.memory.memory_per_class)
        # Distribute idx to all processes
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier()  # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        # print("Label in update_memory: ", label)
        for i, index in enumerate(idx):
            if self.memory.cls_count[label[i].item()] >= self.memory.memory_per_class:
                if index < self.memory.memory_per_class and index != -1:
                    self.memory.replace_data([sample[i], label[i].item()], index)
            ### Until buffer is full, no index is passed in so that samples are simply appended to the buffer
            else:
                self.memory.replace_data([sample[i], label[i].item()])



    # ### Updates memory, just tracking indices and labels
    # def update_memory(self, sample, label):
    #     # print("Updating memory")
    #     # Update memory
    #     if self.distributed:
    #         sample = torch.cat(self.all_gather(sample.to(self.device)))
    #         label = torch.cat(self.all_gather(label.to(self.device)))
    #         sample = sample.cpu()
    #         label = label.cpu()
    #     idx = []
    #     if self.is_main_process():
    #         for lbl in label:
    #             self.memory.seen += 1
    #             # if len(self.memory) < self.memory_size:
    #             if len(self.memory) < self.memory.memory_size:
    #                 idx.append(-1)
    #             else:
    #                 j = torch.randint(0, self.memory.seen, (1, )).item()
    #                 if j < self.memory.memory_size:
    #                     idx.append(j)
    #                 else:
    #                     idx.append(self.memory.memory_size)
    #     # Distribute idx to all processes
    #     if self.distributed:
    #         idx = torch.tensor(idx).to(self.device)
    #         size = torch.tensor([idx.size(0)]).to(self.device)
    #         dist.broadcast(size, 0)
    #         if dist.get_rank() != 0:
    #             idx = torch.zeros(size.item(),
    #                               dtype=torch.long).to(self.device)
    #         dist.barrier()  # wait for all processes to reach this point
    #         dist.broadcast(idx, 0)
    #         idx = idx.cpu().tolist()
    #     # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
    #     print("Label in update_memory: ", label)
    #     for i, index in enumerate(idx):
    #         if len(self.memory) >= self.memory.memory_size:
    #             if index < self.memory.memory_size:
    #                 self.memory.replace_data([sample[i], label[i].item()], index)
    #         ### Until buffer is full, no index is passed in so that samples are simply appended to the buffer
    #         else:
    #             self.memory.replace_data([sample[i], label[i].item()])





    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, None)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()





    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, None)

    def add_new_batch_class(self, class_name):
        batch_exposed_classes = []
        # offset = self.label_offset[self.task_id] if self.per_task_datasets else 0
        for label in class_name:
            if label.item() not in self.batch_exposed_classes:
                self.batch_exposed_classes.append(label.item())
        if self.distributed:
            batch_exposed_classes = torch.cat(
                self.all_gather(
                    torch.tensor(self.batch_exposed_classes,
                                 device=self.device))).cpu().tolist()
            self.batch_exposed_classes = []
            for cls in batch_exposed_classes:
                if cls not in self.batch_exposed_classes:
                    self.batch_exposed_classes.append(cls)
        self.batch_exposed_classes_names = [self.train_dataset.classes_names[i]
                                                for i in self.batch_exposed_classes]

    def add_new_class(self, class_name):
        _old_num = len(self.exposed_classes)
        super().add_new_class(class_name)

        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        if self.memory_size > 0:
            self.batch_exposed_classes = self.exposed_classes
            self.batch_exposed_classes_names = self.exposed_classes_names
        else:
            self.add_new_batch_class(class_name)

    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"Num_Batch_Classes {len(self.batch_exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )
