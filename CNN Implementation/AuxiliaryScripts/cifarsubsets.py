import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import copy
import math


########################################################################################################################################
###     Online Datastreams
########################################################################################################################################


### Rapidly switches domains between cifar and pmnist tasks
def get_online_mixed_cifar_pmnist(seed = 0, set_num=0,setsize=-1, printpath=True):
    """
    Sequence: 
    28 tasks where:
        set_num = 0: CIFAR100-split-1, 
        set_num = 1: PMNIST-0-set-0, 
        set_num = 2: PMNIST-1-set-0, 
        set_num = 3: CIFAR100-split-2, 
        set_num = 4: PMNIST-2-set-0
        set_num = 5: PMNIST-3-set-0
        set_num = 6: CIFAR100-split-3, 
        set_num = 7: PMNIST-4-set-0
        set_num = 8: PMNIST-5-set-0
        set_num = 9: CIFAR100-split-4, 
        set_num = 10: PMNIST-0-set-1
        set_num = 11: PMNIST-1-set-1
        set_num = 12: CIFAR100-split-5, 
        set_num = 13: PMNIST-2-set-1
        set_num = 14: PMNIST-3-set-1
        set_num = 15: CIFAR100-split-6, 
        set_num = 16: PMNIST-4-set-1
        set_num = 17: PMNIST-5-set-1
        set_num = 18: CIFAR100-split-7, 
        set_num = 19: PMNIST-0-set-2
        set_num = 20: PMNIST-1-set-2
        set_num = 21: CIFAR100-split-8, 
        set_num = 22: PMNIST-2-set-2
        set_num = 23: PMNIST-3-set-2
        set_num = 24: CIFAR100-split-9, 
        set_num = 25: PMNIST-4-set-2
        set_num = 26: PMNIST-5-set-2
        set_num = 27: CIFAR100-split-10, 
    """

    data={}
    if set_num % 3 == 0:
        set_num = int(set_num / 3) + 1
        pathx = ('../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/X.pt')
        pathy = ('../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/y.pt')
    elif set_num in [1, 10, 19]:
        set_num = [1, 10, 19].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt")
    elif set_num in [2, 11, 20]:
        set_num = [2, 11, 20].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/1/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/1/" + str(set_num) + "/y.pt")
    elif set_num in [4, 13, 22]:
        set_num = [4, 13, 22].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/2/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/2/" + str(set_num) + "/y.pt")
    elif set_num in [5, 14, 23]:
        set_num = [5, 14, 23].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/3/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/3/" + str(set_num) + "/y.pt")
    elif set_num in [7, 16, 25]:
        set_num = [7, 16, 25].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/4/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/4/" + str(set_num) + "/y.pt")
    elif set_num in [8, 17, 26]:
        set_num = [8, 17, 26].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/5/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/5/" + str(set_num) + "/y.pt")

    if printpath == True:
        # print("Loading from: ",)
        print("Path for X: ", pathx)
        print("Path for Y: ", pathy)
    data['x'] = torch.load(pathx)
    data['y'] = torch.load(pathy)
    return data
    
    


### Not used in the experiments
### Rapidly switches domains between cifar and pmnist tasks
def get_online_simplemixed_cifar_pmnist(seed = 0, set_num=0,setsize=-1, printpath=True):
    """
    Sequence: 
    28 tasks where:
        set_num = 0: CIFAR100-split-1, 
        set_num = 1: PMNIST-0-set-0, 
        set_num = 2: PMNIST-1-set-0, 
        set_num = 3: CIFAR100-split-2, 
        set_num = 4: PMNIST-0-set-0
        set_num = 5: PMNIST-1-set-0
        set_num = 6: CIFAR100-split-3, 
        set_num = 7: PMNIST-0-set-0
        set_num = 8: PMNIST-1-set-0
        set_num = 9: CIFAR100-split-4, 
        set_num = 10: PMNIST-0-set-0
        set_num = 11: PMNIST-1-set-0
        set_num = 12: CIFAR100-split-5, 
        set_num = 13: PMNIST-0-set-0
        set_num = 14: PMNIST-1-set-0
        set_num = 15: CIFAR100-split-6, 
        set_num = 16: PMNIST-0-set-0
        set_num = 17: PMNIST-1-set-0
        set_num = 18: CIFAR100-split-7, 
        set_num = 19: PMNIST-0-set-0
        set_num = 20: PMNIST-1-set-0
        set_num = 21: CIFAR100-split-8, 
        set_num = 22: PMNIST-0-set-0
        set_num = 23: PMNIST-1-set-0
        set_num = 24: CIFAR100-split-9, 
        set_num = 25: PMNIST-0-set-0
        set_num = 26: PMNIST-1-set-0
        set_num = 27: CIFAR100-split-10, 
    """

    data={}
    if set_num % 3 == 0:
        set_num = int(set_num / 3) + 1
        pathx = ('../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/X.pt')
        pathy = ('../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/y.pt')
    elif set_num in [1,4,7, 10,13,16,19]:
        set_num = [1,4,7, 10,13,16,19].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt")
    elif set_num in [2,5,8, 11,14,17, 20]:
        set_num = [2,5,8, 11,14,17, 20].index(set_num)
        pathx = ('../data/PMNIST/Online/' + str(setsize) + "/1/" + str(set_num) + "/X.pt")
        pathy = ('../data/PMNIST/Online/' + str(setsize) + "/1/" + str(set_num) + "/y.pt")

    if printpath == True:
        # print("Loading from: ",)
        print("Path for X: ", pathx)
        print("Path for Y: ", pathy)
    data['x'] = torch.load(pathx)
    data['y'] = torch.load(pathy)
    return data
    
    
























### OMCPR
### Switches between tasks but RotMNIST gradually shifts in 10-degree increments over subsequent sets in which it appears
def get_online_cifar_rotmnist(seed = 0, set_num=0,setsize=-1, printpath=True):
    """
    Sequence: 
    23 tasks where:
        set_num = 0: CIFAR100-split-1, 
        set_num = 1: PMNIST-0-set-0, 
        set_num = 2: RotMNIST-source, 
        set_num = 3: CIFAR100-split-2, 
        set_num = 4: PMNIST-0-set-1
        set_num = 5: RotMNIST-inter-0
        set_num = 6: CIFAR100-split-3, 
        set_num = 7: PMNIST-0-set-2, 
        set_num = 8: RotMNIST-inter-1
        set_num = 9: CIFAR100-split-4, 
        set_num = 10: PMNIST-0-set-3, 
        set_num = 11: RotMNIST-inter-2
        set_num = 12: CIFAR100-split-5, 
        set_num = 13: PMNIST-0-set-4, 
        set_num = 14: RotMNIST-inter-3
        set_num = 15: CIFAR100-split-6, 
        set_num = 16: PMNIST-0-set-5 
        set_num = 17: RotMNIST-inter-4
        set_num = 18: CIFAR100-split-7, 
        set_num = 19: PMNIST-0-set-6, 
        set_num = 20: RotMNIST-inter-5
        set_num = 21: CIFAR100-split-8, 
        set_num = 22: PMNIST-0-set-7, 
    """

    data={}
    print
    if set_num % 3 == 0:
        set_num = int(set_num / 3) + 1
        pathx = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/X.pt'
        pathy = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/y.pt'
    elif set_num %3 == 1:
        set_num = int(math.floor(set_num/3))
        print("Getting PMNIST 0 set number: ", set_num)
        pathx = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt"
        pathy = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt"
    elif set_num % 3 == 2:
        set_num = int(math.floor(set_num/3))
        if set_num == 0:
            print("Getting RotMNIST Source")
            pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/X.pt')
            pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/y.pt')
        else:
            print("Getting RotMNIST Intermediate: ", set_num)
            pathx = ('../data/RotMNIST/Online/' + str(setsize) + "/inter/" + str(set_num-1) + '/0/X.pt')
            pathy = ('../data/RotMNIST/Online/' + str(setsize) + "/inter/" + str(set_num-1) + '/0/y.pt')
    if printpath == True:
            # print("Loading from: ")
            print("Path for X: ", pathx)
            print("Path for Y: ", pathy)

    data['x'] = torch.load(pathx)
    data['y'] = torch.load(pathy)

    # data['x'] = data['x'].type(torch.FloatTensor)
    return data
    
    



### OMCPR-20
### Same as online_cifar_rotmnist but it skips in 20-degree increments of rotation for more drastic distribution shifts of RotMNIST
def get_online_cifar_jump20rotmnist(seed = 0, set_num=0,setsize=-1, printpath=True):
    """
    Sequence: 
    20 tasks where:
        set_num = 0: CIFAR100-split-1, 
        set_num = 1: PMNIST-0-set-0, 
        set_num = 2: RotMNIST-source, 

        set_num = 3: CIFAR100-split-2, 
        set_num = 4: PMNIST-0-set-1

        set_num = 5: CIFAR100-split-3, 
        set_num = 6: PMNIST-0-set-2, 
        set_num = 7: RotMNIST-inter-1

        set_num = 8: CIFAR100-split-4, 
        set_num = 9: PMNIST-0-set-3, 

        set_num = 10: CIFAR100-split-5, 
        set_num = 11: PMNIST-0-set-4, 
        set_num = 12: RotMNIST-inter-3

        set_num = 13: CIFAR100-split-6, 
        set_num = 14: PMNIST-0-set-5 

        set_num = 15: CIFAR100-split-7, 
        set_num = 16: PMNIST-0-set-6, 
        set_num = 17: RotMNIST-inter-5

        set_num = 18: CIFAR100-split-8, 
        set_num = 19: PMNIST-0-set-7, 
    """

    data={}
    if set_num in [0,3,5,8,10,13,15,18]:
        set_num = [0,3,5,8,10,13,15,18].index(set_num)+1
        print("CIFAR Tasknum: ", set_num)
        # set_num = int(set_num / 3) + 1
        pathx = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/X.pt'
        pathy = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/y.pt'
    elif set_num in [1,4,6,9,11,14,16,19]:
        set_num = [1,4,6,9,11,14,16,19].index(set_num)
        # set_num = int(math.floor(set_num/3))
        print("Getting PMNIST 0 set number: ", set_num)
        pathx = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt"
        pathy = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt"
    elif set_num == 2:
        print("Getting RotMNIST Source")
        pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/X.pt')
        pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/y.pt')
    elif set_num == 7:
        print("Getting RotMNIST Intermediate: 1")
        pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/1/0/X.pt')
        pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/1/0/y.pt')
    elif set_num == 12:
        print("Getting RotMNIST Intermediate: 3")
        pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/3/0/X.pt')
        pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/3/0/y.pt')
    elif set_num == 17:
        print("Getting RotMNIST Intermediate: 5")
        pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/X.pt')
        pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/y.pt')
    if printpath == True:
            # print("Loading from: ")
            print("Path for X: ", pathx)
            print("Path for Y: ", pathy)

    data['x'] = torch.load(pathx)
    data['y'] = torch.load(pathy)

    return data
    





### OMCPR-30
### Same as online_cifar_rotmnist but it skips in 30-degree increments of rotation for more drastic distribution shifts of RotMNIST
def get_online_cifar_jump30rotmnist(seed = 0, set_num=0,setsize=-1, printpath=True):
    """
    Sequence: 
    19 tasks where:
        set_num = 0: CIFAR100-split-1, 
        set_num = 1: PMNIST-0-set-0, 
        set_num = 2: RotMNIST-source,
        
        set_num = 3: CIFAR100-split-2, 
        set_num = 4: PMNIST-0-set-1
        
        set_num = 5: CIFAR100-split-3, 
        set_num = 6: PMNIST-0-set-2, 
        
        set_num = 7: CIFAR100-split-4, 
        set_num = 8: PMNIST-0-set-3, 
        set_num = 9: RotMNIST-inter-2
        
        set_num = 10: CIFAR100-split-5, 
        set_num = 11: PMNIST-0-set-4, 
        
        set_num = 12: CIFAR100-split-6, 
        set_num = 13: PMNIST-0-set-5 
        
        set_num = 14: CIFAR100-split-7, 
        set_num = 15: PMNIST-0-set-6, 
        set_num = 16: RotMNIST-inter-5
        
        set_num = 17: CIFAR100-split-8, 
        set_num = 18: PMNIST-0-set-7, 
    """

    data={}
    if set_num in [0,3,5,7,10,12,14,17]:
        set_num = [0,3,5,7,10,12,14,17].index(set_num)+1
        print("CIFAR Tasknum: ", set_num)
        pathx = '../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/X.pt'
        pathy = '../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/y.pt'
    elif set_num in [1,4,6,8,11,13,15,18]:
        set_num = [1,4,6,8,11,13,15,18].index(set_num)
        print("PMNIST Tasknum: ", set_num)
        print("Getting PMNIST 0 set number: ", set_num)
        pathx = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt"
        pathy = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt"
    elif set_num == 2:
        set_num = int(math.floor(set_num/3))
        print("Getting RotMNIST Source")
        pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/X.pt')
        pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/y.pt')
    elif set_num == 9:
        print("Getting RotMNIST Intermediate: 2")
        pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/2/0/X.pt')
        pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/2/0/y.pt')
    elif set_num == 16:
        print("Getting RotMNIST Intermediate: 5")
        pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/X.pt')
        pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/y.pt')
    if printpath == True:
            # print("Loading from: ")
            print("Path for X: ", pathx)
            print("Path for Y: ", pathy)

    data['x'] = torch.load(pathx)
    data['y'] = torch.load(pathy)

    return data
    




















































































































def make_splitcifar(seed=0, pc_valid=0.2):
    data={}
    taskcla=[]
    size=[3,32,32]
    
    
    # CIFAR10
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    
    # CIFAR10
    dat={}
    dat['train']=datasets.CIFAR10('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR10('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='cifar10'
    data[0]['ncla']=10
    data[0]['train']={'x': [],'y': []}
    data[0]['test']={'x': [],'y': []}
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            data[0][s]['x'].append(image)
            data[0][s]['y'].append(target.numpy()[0])
    
    # "Unify" and save
    for s in ['train','test']:
        data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
        data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
    
    # CIFAR100
    dat={}
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    for n in range(1,11):
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']=10
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            task_idx = target.numpy()[0] // 10 + 1
            data[task_idx][s]['x'].append(image)
            data[task_idx][s]['y'].append(target.numpy()[0]%10)
    
    
    
    for t in range(1,11):
        for s in ['train','test']:
            data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
            data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
            
    os.makedirs('../data/split_cifar/' ,exist_ok=True)
    
    for t in range(0,11):
      # Validation
      r=np.arange(data[t]['train']['x'].size(0))
      r=np.array(shuffle(r,random_state=seed),dtype=int)
      nvalid=int(pc_valid*len(r))
      ivalid=torch.LongTensor(r[:nvalid])
      itrain=torch.LongTensor(r[nvalid:])
      data[t]['valid']={}
      data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
      data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
      data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
      data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
    
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/split_cifar/' + str(t)) ,exist_ok=True)
        torch.save(data[t][s]['x'], ('../data/split_cifar/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(data[t][s]['y'], ('../data/split_cifar/'+ str(t) + '/y_' + s + '.bin'))
    
    

def make_PMNIST(seed=0, pc_valid=0.1):
    
    mnist_train = datasets.MNIST('../data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),transforms.Resize((32,32))]), download = True)        
    mnist_test = datasets.MNIST('../data/', train = False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),transforms.Resize((32,32))]), download = True)        

    # Create a dictionary to store the converted dataset
    # converted_train = {'data': [], 'labels': []}
    # Create a dictionary to store the converted dataset
    # converted_test = {'data': [], 'labels': []}
    
    # # Convert and store the dataset
    # for idx, (image, label) in enumerate(mnist_train):
    #     # Convert the image to 3-channel RGB-like format
    #     # image_rgb = convert_to_rgb(image.unsqueeze(0))  # Convert to batch format with one image
        
    #     converted_train['data'].append(image)
    #     converted_train['labels'].append(label)

    # Convert and store the dataset
    # for idx, (image, label) in enumerate(mnist_test):
    #     # Convert the image to 3-channel RGB-like format
    #     # image_rgb = convert_to_rgb(image.unsqueeze(0))  # Convert to batch format with one image
        
    #     converted_test['data'].append(image)
    #     converted_test['labels'].append(label)
    
    # # Convert the lists to PyTorch tensors
    # converted_train['data'] = torch.stack(converted_train['data'])
    # # converted_train['labels'] = torch.tensor(converted_train['labels'])
    # converted_test['data'] = torch.stack(converted_test['data'])
    # # converted_test['labels'] = torch.tensor(converted_test['labels'])

    # converted_train['data'] = mnist_train.data
    # converted_train['labels'] = mnist_train.targets

    # converted_test['data'] = mnist_test.data
    # converted_test['labels'] = mnist_test.targets


    dat={}
    data={}
    taskcla=[]
    size=[1,32,32]    
    os.makedirs('../data/PMNIST', exist_ok =True)
    
    dat['train']=mnist_train
    dat['test']=mnist_test
    
    ### Prepare the data variable and lists of label indices for further processing
    for t in range(0,6):
      data[t]={}
      data[t]['name']='PMNIST'
      data[t]['ncla']=10
      data[t]['train']={'x': [],'y': []}
      data[t]['test']={'x': [],'y': []}



    for t in range(0,6):
        torch.manual_seed(t)
        taskperm = torch.randperm((32*32))
        # ### Extract only the appropriately labeled samples for each of the subsets
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            ### For each image we will flatten it, permute it according to taskperm, and then reshape it and convert it to produce (3,32,32) image shape
            for image,target in loader:   
            # for batch in loader:
            #     image = batch[0]  # Extract the image from the batch
            #     target = batch[1]  # Extract the target from the batch

                ### Flatten the (1,32,32) image into (1,1024)
                image = torch.flatten(image)
                # print("Image shape: ", image.shape)
                image = image[taskperm]
                image = image.view(1,32,32)
                ### Should give shape (3,32,32)
                image = torch.cat((image,image,image), dim=0)

                data[t][s]['x'].append(image)
                data[t][s]['y'].append(target.numpy()[0])
      
            data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,3,32,32)
            data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
    #!# Removed some processing steps
    
    ### Splitting validation off from training rather than test is fine here, so long as both sets are preprocessed identically
    for t in range(0,6):
      # Validation
      torch.manual_seed(t)
      taskperm = torch.randperm(data[t]['train']['x'].size(0))

      nvalid=int(pc_valid*len(taskperm))
      ivalid=torch.LongTensor(taskperm[:nvalid])
      itrain=torch.LongTensor(taskperm[nvalid:])
      data[t]['valid']={}
      data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
      data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
      ### Only overwrites the train key after its been used to create the valid subset
      data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
      data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/PMNIST/' + str(t)) ,exist_ok=True)
        torch.save(data[t][s]['x'], ('../data/PMNIST/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(data[t][s]['y'], ('../data/PMNIST/'+ str(t) + '/y_' + s + '.bin'))
    
    
    
    
def make_subsets(seed=0, pc_valid=0.1):
        
    dat={}
    data={}
    taskcla=[]
    size=[3,32,32]    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    train_labels = [[],[],[],[],[],[]]
    test_labels = [[],[],[],[],[],[]]
    ### Keys indicating which fine labels from CIFAR100 will be included in each of the data subsets
    keys = [["bicycle", "bus", "motorcycle", "pickup_truck","train","lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            ["beaver","dolphin","otter","seal","whale","aquarium_fish","flatfish","ray", "shark","trout"],
            ["bed","chair","couch","table","wardrobe","clock","keyboard","lamp","telephone","television"],
            ["hamster","mouse","rabbit","shrew","squirrel","fox","porcupine","possum","raccoon","skunk"],
            ["orchid","poppy","rose","sunflower","tulip","apple","mushroom","orange","pear","sweet_pepper"],
            ["crab","lobster","snail","spider","worm","bee","beetle","butterfly","caterpillar","cockroach"]]
    
    os.makedirs('../data/split_cifar100', exist_ok =True)
    
    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    ### Prepare the data variable and lists of label indices for further processing
    for t in range(0,6):
      data[t]={}
      data[t]['name']='cifar100'
      data[t]['ncla']=10
      data[t]['train']={'x': [],'y': []}
      data[t]['test']={'x': [],'y': []}
      train_labels[t] = [dat['train'].class_to_idx[k] for k in keys[t]]
      test_labels[t] = [dat['test'].class_to_idx[k] for k in keys[t]]
    
    ### Extract only the appropriately labeled samples for each of the subsets
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:      
          for t in range(0,6):
            if target in (train_labels[t]):
              data[t][s]['x'].append(image)
              data[t][s]['y'].append(target.numpy()[0])
          
    # Deep copy the nested list
    dataprocessed = copy.deepcopy(data)
    
    ### Convert the labels of each subset to range from 0-9
    for t in range(0,6):
      for l in range(0,10):
        for i in range(len(data[t]["train"]["y"])):
          if data[t]["train"]["y"][i] == train_labels[t][l]:
            dataprocessed[t]["train"]["y"][i] = l
    
        for i in range(len(data[t]["test"]["y"])):
          if data[t]["test"]["y"][i] == test_labels[t][l]:
            dataprocessed[t]["test"]["y"][i] = l
    
    os.makedirs("../data/sim_cifar100",exist_ok=True)
    
    for t in range(0,6):
        for s in ['train','test']:
            dataprocessed[t][s]['x']=torch.stack(dataprocessed[t][s]['x']).view(-1,size[0],size[1],size[2])
            dataprocessed[t][s]['y']=torch.LongTensor(np.array(dataprocessed[t][s]['y'],dtype=int)).view(-1)

    
    ### Splitting validation off from training rather than test is fine here, so long as both sets are preprocessed identically
    for t in range(0,6):
      # Validation
      r=np.arange(dataprocessed[t]['train']['x'].size(0))
      r=np.array(shuffle(r,random_state=seed),dtype=int)
      nvalid=int(pc_valid*len(r))
      ivalid=torch.LongTensor(r[:nvalid])
      itrain=torch.LongTensor(r[nvalid:])
      dataprocessed[t]['valid']={}
      dataprocessed[t]['valid']['x']=dataprocessed[t]['train']['x'][ivalid].clone()
      dataprocessed[t]['valid']['y']=dataprocessed[t]['train']['y'][ivalid].clone()
      dataprocessed[t]['train']['x']=dataprocessed[t]['train']['x'][itrain].clone()
      dataprocessed[t]['train']['y']=dataprocessed[t]['train']['y'][itrain].clone()
    
      for s in ['train','valid','test']:
        os.makedirs(('../data/sim_cifar100/' + str(t)) ,exist_ok=True)
        torch.save(dataprocessed[t][s]['x'], ('../data/sim_cifar100/'+ str(t) + '/x_' + s + '.bin'))
        torch.save(dataprocessed[t][s]['y'], ('../data/sim_cifar100/'+ str(t) + '/y_' + s + '.bin'))
    
    




def make_cifar100full(seed=0, pc_valid=0.2):
    data={}
    taskcla=[]
    size=[3,32,32]
    
    
    # CIFAR100
    dat={}
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    
    # CIFAR10
    dat={}
    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='cifar100'
    data[0]['ncla']=100
    data[0]['train']={'x': [],'y': []}
    data[0]['test']={'x': [],'y': []}
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            data[0][s]['x'].append(image)
            data[0][s]['y'].append(target.numpy()[0])
    
    # "Unify" and save
    for s in ['train','test']:
        data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
        data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
    
    os.makedirs('../data/cifar100full/' ,exist_ok=True)
    
    # Validation
    r=np.arange(data[0]['train']['x'].size(0))
    r=np.array(shuffle(r,random_state=seed),dtype=int)
    nvalid=int(pc_valid*len(r))
    ivalid=torch.LongTensor(r[:nvalid])
    itrain=torch.LongTensor(r[nvalid:])
    data[0]['valid']={}
    data[0]['valid']['x']=data[0]['train']['x'][ivalid].clone()
    data[0]['valid']['y']=data[0]['train']['y'][ivalid].clone()
    data[0]['train']['x']=data[0]['train']['x'][itrain].clone()
    data[0]['train']['y']=data[0]['train']['y'][itrain].clone()

    
    for s in ['train','valid','test']:
        torch.save(data[0][s]['x'], ('../data/cifar100full/' + '/x_' + s + '.bin'))
        torch.save(data[0][s]['y'], ('../data/cifar100full/' + '/y_' + s + '.bin'))
        