from __future__ import print_function
import os
import argparse
import time
import torch
torch.set_printoptions(threshold=5000)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.insert(0,'../data_prep/data_loading')
import data_classes
from torchvision import datasets, models, transforms
from utility_functions import initialize_model
import pickle

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    
    model.train()

    for batch_idx, (data,target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        
        if not args.no_train_log:
            if batch_idx % args.log_interval == 0:
                print_out('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),args.print_log)                

def test(args, model, device, test_loader, criterion, epoch):
    #pdb.set_trace()
    model.eval()
    test_loss = 0
    correct = 0
    indiv_acc_dict = {}
    for i in range(args.num_classes):
        indiv_acc_dict[i] = [0,0,0]
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            correct += pred.eq(target.view_as(pred)).sum().item()

           
            for i in range(len(pred)):
                indiv_acc_dict[int(pred[i])][2] += 1          
                indiv_acc_dict[int(target.view_as(pred)[i])][0] += 1
                if int(pred[i]) == int(target.view_as(pred)[i]):
                    indiv_acc_dict[int(pred[i])][1] += 1
    if not args.no_outputs:
        torch.save(output, os.path.join('../outputs',args.outputdir,'%s_output.pt'%str(epoch)))
        torch.save(target, os.path.join('../outputs',args.outputdir,'%s_target.pt'%str(epoch)))
                    
    test_loss /= len(test_loader.dataset)
    if args.no_train_log:
        print_out('epoch: ' + str(epoch),args.print_log)
    print_out('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),args.print_log)
    if not args.no_indiv_accuracies:
        print_out('class    total     guessed    accuracy    f1-score',args.print_log)
        for prednum in indiv_acc_dict:
            if indiv_acc_dict[prednum][0] == 0:
                print_out('no samples for class %s'%str(prednum+1),args.print_log)
            else:
                if not args.include_dotless:
                    numclass = prednum+1
                else:
                    numclass = prednum
                total = indiv_acc_dict[prednum][0]
                guessed = indiv_acc_dict[prednum][2]
                accuracy = round(indiv_acc_dict[prednum][1]/indiv_acc_dict[prednum][0],3)
                f1 = round(total*accuracy/(total+guessed)*2,3)
                print_out('%s        %s         %s        %s         %s'%(str(numclass),str(total),str(guessed),str(accuracy),str(f1)),args.print_log)
                indiv_acc_dict[prednum].append(','.join([str(round(accuracy,3)),str(guessed),str(round(f1,3))]))
    if not args.no_outputs:
        output_string = indiv_acc_dict[0][3]
        for i in range(args.num_classes):
            output_string += ','+indiv_acc_dict[i][3]
        args.testing_log.write(output_string+'\n')
        args.testing_log.flush()

def print_out(text,log):
    print(text)
    log.write(str(text))
    log.flush()


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    #load model
    model, input_size, params_to_update = initialize_model(args.model, args.num_classes, args.feature_extract, use_pretrained=args.pretrained)

    model.to(device)
    if args.multi_gpu:
        model = nn.DataParallel(model)

    #data loading
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        data_classes.CountingDotsDataSet(root_dir='../stimuli/'+args.input_data,num_classes=args.num_classes, train=True,include_dotless=args.include_dotless,transform=data_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        data_classes.CountingDotsDataSet(root_dir='../stimuli/%s'%args.input_data,num_classes=args.num_classes, train=False,include_dotless=args.include_dotless,transform=data_transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
  
    #Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params_to_update, lr=args.lr)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)
        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum)

    #loss
    criterion = nn.CrossEntropyLoss()


    #output directory
    if args.feature_extract:
        args.outputdir = '%s_%s_classes%s_featureextract_%s'%(args.input_data.replace('/','_'),args.model.split('.')[-1],args.num_classes,time.strftime('%m-%d-%Y:%H_%M'))
    elif args.pretrained:
        args.outputdir = '%s_%s_classes%s_pretrained_%s'%(args.input_data.replace('/','_'),args.model.split('.')[-1],args.num_classes,time.strftime('%m-%d-%Y:%H_%M'))
    else:
        args.outputdir = '%s_%s_classes%s_%s'%(args.input_data.replace('/','_'),args.model.split('.')[-1],args.num_classes,time.strftime('%m-%d-%Y:%H_%M'))

    if not args.no_outputs:
        if not os.path.exists(os.path.join('../outputs',args.outputdir)):
            os.mkdir(os.path.join('../outputs',args.outputdir))
        args.testing_log = open(os.path.join('../outputs',args.outputdir,'testing_log.csv'),'w+')
        for i in range(args.min_label,args.num_classes+1):
             args.testing_log.write('%s acc,%s guessed,%s F1'%(str(i),str(i),str(i)))
             if i!=args.num_classes:
                args.testing_log.write(',')
        args.testing_log.write('\n')
        args.testing_log.flush()
        args_file = open(os.path.join('../outputs',args.outputdir,'args.txt'),'w+')
        args_file.write(str(args))
        args_file.close()



    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, test_loader, criterion, epoch)
        if epoch%args.save_model_interval == 0:
            torch.save(model,os.path.join('../outputs',args.outputdir,'model_%s.pt'%str(epoch)))
        
if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='How Many Dots? Network Training')
    parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='TBS',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--model', type=str, default='resnet', metavar='M',
                        help='neural net model to use (default: resnet, other options: alexnet, vgg, squeezenet, densenet)')
    parser.add_argument('--num-classes', type=int, default=20, metavar='C',
                        help='max number of label, all higher numbers will be binned to num_classes (default: 20, all correct labels)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='initialize model at trained ImageNet weights')
    parser.add_argument('--feature-extract', action='store_true', default=False,
                        help='do not train the whole network just the last classification layer. Will automatically set "pretrained" to true.')        
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                        help='optimization algorithm to use (default: SGD other options: adam)')   
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--momentum', type=float, default=0.2, metavar='m',
                        help='SGD momentum (default: 0.2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-indiv-accuracies', action='store_true', default=False,
                        help='Dont print individual accuracies for each class during testing')
    parser.add_argument('--no-train-log', action='store_true', default=False,
                        help='supressing the training print out so you only see testing')
    parser.add_argument('--no-outputs', action='store_true', default=False,
                        help='dont save the outputs of the network')
    parser.add_argument('--input-data', type=str, default='enumeration', metavar='I',
                        help='input folder name to use for input data (default: enumeration)')
    parser.add_argument('--include-dotless', action='store_true', default=False,
                        help='The dataset has images with label "0" with no dots in the image' )        
    parser.add_argument('--save-model-interval', type=int, default=5, metavar='SM',
                        help='Every time the epoch number is divisible by this number, save the model (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='W',
                        help='number of parallel batches to process. Rule of thumb 4*num_gpus (default: 4)')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='run model on multiple gpus')    
 

    args = parser.parse_args()
    # reconcile arguments
    if args.feature_extract:
        args.pretrained == True


    #output directory
    if args.feature_extract:
        args.outputdir = '%s_%s_classes%s_featureextract_%s'%(args.input_data.replace('/','_'),args.model.split('.')[-1],args.num_classes,time.strftime('%m-%d-%Y:%H_%M'))
    elif args.pretrained:
        args.outputdir = '%s_%s_classes%s_pretrained_%s'%(args.input_data.replace('/','_'),args.model.split('.')[-1],args.num_classes,time.strftime('%m-%d-%Y:%H_%M'))
    else:
        args.outputdir = '%s_%s_classes%s_%s'%(args.input_data.replace('/','_'),args.model.split('.')[-1],args.num_classes,time.strftime('%m-%d-%Y:%H_%M'))
    if not os.path.exists(os.path.join('../outputs',args.outputdir)):
        os.mkdir(os.path.join('../outputs',args.outputdir))

    args.print_log = open(os.path.join('../outputs',args.outputdir,'print_log.txt'),'w+') #file logs everything that prints to console


    print_out('running with args:',args.print_log)
    print_out(args,args.print_log)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print_out('using cuda',args.print_log)
    if args.include_dotless:
        args.min_label = 0
    else:
        args.min_label = 1
    main(args)

    print_out('Total Run Time:',args.print_log)
    print_out("--- %s seconds ---" % (time.time() - start_time),args.print_log)

