import numpy as np
import torch
import torch_optimizer
from torch.nn import Module
from torch import optim
from torch.optim import lr_scheduler
import copy


acts = {}


def set_hook(modality, block, adapter, name):
    def hook_fn(module, input, output):
        acts[modality][block][adapter][name] = output.detach().cpu()
    return hook_fn


def set_all_layers(clip_model, hook_handles, subnet, only_subnet_adapters):    
    acts['text'] = {}
    for block_idx, block in enumerate(clip_model.transformer.resblocks):
        acts['text'][block_idx] = {}
        for adapter_idx, adapter in enumerate(block.adaptmlp_list):
            ### If we are only gathering activations for the subnet's dedicated adapters, otherwise store all activations
            if only_subnet_adapters and adapter_idx not in block.frozen_experts[subnet]:
                continue
            else:
                acts['text'][block_idx][adapter_idx] = {}
                hook_handles.append(adapter.down_proj.register_forward_hook(set_hook('text', block_idx, adapter_idx, "down_proj")))
                hook_handles.append(adapter.up_proj.register_forward_hook(set_hook('text', block_idx, adapter_idx, "up_proj")))



    acts['visual'] = {}
    for block_idx, block in enumerate(clip_model.visual.transformer.resblocks):
        acts['visual'][block_idx] = {}
        for adapter_idx, adapter in enumerate(block.adaptmlp_list):
            if only_subnet_adapters and adapter_idx not in block.frozen_experts[subnet]:
                continue
            else:
                acts['visual'][block_idx][adapter_idx] = {}
                hook_handles.append(adapter.down_proj.register_forward_hook(set_hook('visual', block_idx, adapter_idx, "down_proj")))
                hook_handles.append(adapter.up_proj.register_forward_hook(set_hook('visual', block_idx, adapter_idx, "up_proj")))



def store_acts(all_acts, all_labels, y_label, step):
    all_labels = torch.cat((all_labels, y_label.detach().cpu()), dim=0)
    if step == 0:
        for modal in ['text', 'visual']:
            for block_idx in acts[modal].keys():
                for adapter_idx in acts[modal][block_idx].keys():
                    for layer in ['down_proj', 'up_proj']:
                        ### Note: this assumes linear layer outputs, as it is used with the adapter layers
                        all_acts[modal][block_idx][adapter_idx][layer] = acts[modal][block_idx][adapter_idx][layer]



    else: 
        for modal in ['text', 'visual']:
            for block_idx in acts[modal].keys():
                for adapter_idx in acts[modal][block_idx].keys():
                    for layer in ['down_proj', 'up_proj']:
                        all_acts[modal][block_idx][adapter_idx][layer] = torch.cat((all_acts[modal][block_idx][adapter_idx][layer],
                                                                            acts[modal][block_idx][adapter_idx][layer]), dim=0)

    return all_acts, all_labels



### Takes a clip model and returns all activations for a set of data
def activations(clip_model, text_tokens, data_loader, subnet=None, only_subnet_adapters=True):
    print("\n\n\n\n\n\n")
    handles     = []
    all_labels = torch.empty(0)

    set_all_layers(clip_model, handles, subnet, only_subnet_adapters)

    ### Make a copy of the empty acts dict to store values for all batches
    all_acts = copy.deepcopy(acts)




    ### Note: The stacking will cause an error if the dataloader only has one batch, seemingly. Not currently an issue, but leaving this just in case
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data
            clip_model(x_input.cuda(), text_tokens)

            all_acts, all_labels = store_acts(all_acts, all_labels, y_label, step)


    print("Resulting shape of labels: ", all_labels.shape)
    first_expert = list(all_acts['visual'][0].keys)[0]
    print("Resulting shape of one key: ", all_acts['visual'][0][first_expert]['down_proj'].shape)


    for block in all_acts['visual'].keys():
        print("Adapter keys for visual block {}: {}".format(block, all_acts['visual'][block].keys()))


    for handle in handles:
        handle.remove()    

    return handles, all_labels





















































