import os
import torch

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class
from DatasetGenerator import DatasetGenerator
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
import metrics as metrics
from utils import create_logger
from utils import calculate_classwise_accuracy
from timeit import default_timer as timer
import argparse

#Hyperparameters
# lr = 2e-4

threshold = 0.5


def save_model(models, optimizer, scheduler, epoch, args, folder="saved_models/", name="best"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    state = {'epoch': epoch + 1,
             'model_rep': models.state_dict(),
             'optimizer_state': optimizer.state_dict(),
             'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
             'args': vars(args)}

    run_name = "debug" if args.debug else wandb.run.name
    torch.save(state, f"{folder}{args.label}_{run_name}_{name}_model.pkl")


def train(args, debug = False):
    """Training Function"""
    device = ("cuda:2" if torch.cuda.is_available() else "cpu")
    NIH_CLASS_CNT = 20
    IMG_SIZE = 512
    tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]     
    
    logger = create_logger('Main')

    if args.model== "densenet":
        model = DenseNet121_Multi_Class(classCount=NIH_CLASS_CNT)
    elif args.model== "inception":
        model = Inception_Multi_Class(classCount=NIH_CLASS_CNT)
    elif args.model== "resnet":
        model = ResNet_Multi_Class(classCount=NIH_CLASS_CNT)    
    elif args.model== "resnext":
        model = ResNeXt_Multi_Class(classCount=NIH_CLASS_CNT)            
        # print(model)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.7, patience=5)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.2)
    criterion = nn.CrossEntropyLoss()

    if not debug:
        wandb.init(project="nih_lt_benchmark", group=args.label, config=args, reinit=True)
        dropout_str = "" if not args.dropout else "-dropout"
        wandb.run.name = f"nih_lt_{args.model}{dropout_str}-lr:{args.lr}-wd:{args.weight_decay}_" + wandb.run.name

    
    # accum_iter = 64
    # if pretrained:
    #     config = torch.load(model_path)
    #     model.load_state_dict(config['state_dict'])
    #     optimizer.load_state_dict(config['optimizer_state_dict'])

    # nih_classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                # 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'Normal']
    nih_pathDirData = "/data/home1/arunsg/data-pruning/dataset/images"

    nih_pathFileTrain = "/data/home1/arunsg/data-pruning/nih-cxr/nih-cxr-lt_single-label_train.csv"
    nih_pathFileVal = "/data/home1/arunsg/data-pruning/nih-cxr/nih-cxr-lt_single-label_balanced-val.csv"
    nih_pathFileBalancedTest = "/data/home1/arunsg/data-pruning/nih-cxr/nih-cxr-lt_single-label_balanced-test.csv"
    model_storage = "/data/home1/arunsg/gitproject/data-pruning/base_model/"
    
    nih_trBatchSize = args.batch_size
    nih_valBatchSize = args.batch_size

    if debug:
        print("M4: NIH dataset: Loading from dir: ", nih_pathDirData,
                " using training txt: ", nih_pathFileTrain,
                " and validation txt: ", nih_pathFileVal,
                " using test txt: ", nih_pathFileBalancedTest)
    m4_train_data = DatasetGenerator(pathImageDirectory = nih_pathDirData,
                                        pathDatasetFile = nih_pathFileTrain,
                                        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

    if debug:
        print("M4: NIH dataset: Initializing for validation")
    m4_valid_data = DatasetGenerator(pathImageDirectory = nih_pathDirData,
                                        pathDatasetFile = nih_pathFileVal,
                                        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
    
    if debug:
        print("M4: NIH dataset: Initializing for test")
    m4_test_data = DatasetGenerator(pathImageDirectory = nih_pathDirData,
                                        pathDatasetFile = nih_pathFileBalancedTest,
                                        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
        

    
    if debug:
        print("M4: NIH dataset: Dataloader for training")
    nih_dataLoaderTrain = DataLoader(dataset = m4_train_data,
                                    batch_size = nih_trBatchSize,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False) # earlier was true
    if debug:
        print("M4: NIH dataset: Dataloader for validation")
    nih_dataLoaderVal = DataLoader(dataset = m4_valid_data,
                                    batch_size = nih_valBatchSize,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False) # earlier was true
    if debug:
        print("M4: NIH dataset: Dataloader for test")
    nih_dataLoaderTest = DataLoader(dataset = m4_test_data,
                                    batch_size = nih_valBatchSize,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False) # earlier was true

    if debug:
        print("m4 data loaded")       

    
    iter_epochs = tqdm(range(args.num_epochs))
    max_valid_acc = 0.0
    training_samples_cnt = [0] * NIH_CLASS_CNT

    for epoch_num in iter_epochs:
        true_class = torch.zeros(20)
        predict_class = torch.zeros(20)
        train_losses = 0
        train_correct = 0
        valid_losses = 0
        valid_correct = 0
        test_losses = 0
        test_correct = 0
        # training mode
        model.train()
        for batch_no, (images, labels) in enumerate(tqdm(nih_dataLoaderTrain)):
            start = timer()
            images = images.to(device)
            labels = labels.to(device)
            # zeroing the optimizer
            optimizer.zero_grad()
            
            outputs = model(images)
            # prediction = (outputs >= threshold).to(torch.float32)
            # print(outputs.shape)
            # print(labels.shape)
            # exit()
        
            _, labels_index = labels.max(1)
            loss = criterion(outputs, labels_index)   
            
            _, prediction = outputs.max(1)

            if epoch_num == 0:
                for i in range(NIH_CLASS_CNT):
                    training_samples_cnt[i] += torch.sum(labels_index==i).item()
            
            # print(training_samples_cnt)

            # for i in range(20):
                # true_class[i] += torch.sum(labels==i).item()
                # predict_class[i] += torch.sum((labels==i) & (labels == prediction)).item()

            train_losses += loss.item()
            # calculating the gradients
            loss.backward()
            optimizer.step()

            #prediction = prediction.squeeze(axis = 1)
            # labels = labels.squeeze(axis = 1)
            # Correct predictions
            train_correct += (prediction == labels_index).sum()

        train_loss = train_losses / len(nih_dataLoaderTrain.dataset) 
 

        true_class = torch.zeros(20)
        predict_class = torch.zeros(20)

        model.eval()
        # for batch_no, (images, labels) in enumerate(tqdm(nih_dataLoaderVal)):
        for batch_no, (images, labels) in enumerate(nih_dataLoaderVal):
            # print('batch no',batch_no,labels.shape,images.shape)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
           
            _, prediction = outputs.max(1)
            _, labels_index = labels.max(1)

            loss = criterion(outputs, labels_index)

            for i in range(20):
                true_class[i] += torch.sum(labels_index==i).item()
                predict_class[i] += torch.sum((labels_index==i) & (labels_index == prediction)).item()

            
            valid_losses += loss.item()

            valid_correct += (prediction == labels_index).sum()
            
            
        valid_accuracy = valid_correct.item() / len(nih_dataLoaderVal.dataset)
        valid_loss = valid_losses / len(nih_dataLoaderVal.dataset)
        acc = torch.div(predict_class,true_class)

        epoch_stats = {}
        for i in range(20):
            epoch_stats['val_class_acc_'+str(i)] = acc[i].item()

        
        # training_samples_cnt
        li = [i for i in zip(training_samples_cnt,acc.tolist())]
        li = sorted(li, key = lambda i: i[0],reverse=True)

        accuracy = calculate_classwise_accuracy(li)

        print('valid - ',li,accuracy)
        epoch_stats['val_head_acc'] = accuracy['head']
        epoch_stats['val_medium_acc'] = accuracy['medium']
        epoch_stats['val_tail_acc'] = accuracy['tail']
        epoch_stats['val_total_acc'] = accuracy['total']
        



        for batch_no, (images, labels) in enumerate(nih_dataLoaderTest):
            # print('batch no',batch_no,labels.shape,images.shape)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
           
            _, prediction = outputs.max(1)
            _, labels_index = labels.max(1)

            loss = criterion(outputs, labels_index)

            for i in range(20):
                true_class[i] += torch.sum(labels_index==i).item()
                predict_class[i] += torch.sum((labels_index==i) & (labels_index == prediction)).item()

            
            test_losses += loss.item()

            test_correct += (prediction == labels_index).sum()
            
            
        test_accuracy = test_correct.item() / len(nih_dataLoaderTest.dataset)
        test_loss = test_losses / len(nih_dataLoaderTest.dataset)
        acc = torch.div(predict_class,true_class)

        # epoch_stats = {}
        for i in range(20):
            epoch_stats['test_class_acc_'+str(i)] = acc[i].item()

        
        # training_samples_cnt
        li = [i for i in zip(training_samples_cnt,acc.tolist())]
        li = sorted(li, key = lambda i: i[0],reverse=True)

        accuracy = calculate_classwise_accuracy(li)

        print('test - ',li,accuracy)
        epoch_stats['test_head_acc'] = accuracy['head']
        epoch_stats['test_medium_acc'] = accuracy['medium']
        epoch_stats['test_tail_acc'] = accuracy['tail']
        epoch_stats['test_total_acc'] = accuracy['total']
        
        epoch_stats['train_loss'] = train_loss 
        epoch_stats['validation_loss'] = valid_loss
        epoch_stats['test_loss'] = test_loss


        # print('-----------------------------------')
        # print(true_class,predict_class)
        # print(acc)
        # print('-----------------------------------')

        
        iter_epochs.set_description(desc = 'Train Loss {} Validation : Loss {}, Accuracy {} Test : Loss {}, Accuracy {}'.format(train_loss, valid_loss, valid_accuracy, test_loss, test_accuracy))
        
        scheduler.step(valid_loss)


        if not args.debug:
            wandb.log(epoch_stats, step=epoch_num)
        print('-----------------------------------------------------------------------------')
        print('epoch_stats - ',epoch_stats)
        print('-----------------------------------------------------------------------------')
        

        # Any time one of the model_saver metrics is improved upon, store a corresponding model.
        if max_valid_acc < acc.mean().item():
                max_valid_acc = acc.mean().item()
                # Evaluate the model on the test set and store relative results.
                # test_evaluator(args, test_loader, tasks, DEVICE, model, loss_fn, metric, aggregators, logger, k, epoch)
                if args.store_models:
                    # Save (overwriting) any model that improves the average metric
                    save_model(model, optimizer, scheduler, epoch_num, args,
                               folder=model_storage, name="best")

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    # Save training/validation results.
    if args.store_models and (not args.time_measurement_exp):
        # Save last model.
        save_model(model, optimizer, scheduler, epoch_num, args, folder=model_storage,
                   name="last")    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnext', help='model to train')
    parser.add_argument('--label', type=str, default='orig_wp', help='wandb group')
    parser.add_argument('--dataset', type=str, default='nih', help='which dataset to use',
                        choices=['nih'])
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--p', type=float, default=0.1, help='Task dropout probability')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Epochs to train for.')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode: disables wandb.')
    parser.add_argument('--store_models', action='store_true', help='Whether to store  models at fixed frequency.')
    parser.add_argument('--decay_lr', action='store_true', help='Whether to decay the lr with the epochs.')
    parser.add_argument('--dropout', action='store_true', help='Whether to use additional dropout in training.')
    parser.add_argument('--no_dropout', action='store_true', help='Whether to not use dropout at all.')
    parser.add_argument('--store_convergence_stats', action='store_true',
                        help='Whether to store the squared norm of the unitary scalarization gradient at that point')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization.')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of experiment repetitions.')
    parser.add_argument('--random_seed', type=int, default=1, help='Start random seed to employ for the run.')
    
    parser.add_argument('--baseline_losses_weights', type=int, nargs='+',
                        help='Weights to use for losses. Be sure that the ordering is correct! (ordering defined as in config for losses.')
    parser.add_argument('--time_measurement_exp', action='store_true',
                        help="whether to only measure time (does not log training/validation losses/metrics)")
    args = parser.parse_args()
    
    for i in range(args.n_runs):
        train(args, args.debug)
