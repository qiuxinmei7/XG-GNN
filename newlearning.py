import torch
from util import Logger, accuracy
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
import copy
from torch.optim import lr_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from dataloader import BatchImageGenerator
from copy import deepcopy
from typing import List
from datetime import datetime
from tensorboardX import SummaryWriter
import time
class Meta_Train:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder,args,repeat_time) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.optimizers = optimizers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.setup_path(repeat_time)
        self.metainner_optim = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,weight_decay=0.0001
        )

        self.metaouter_optim = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,weight_decay=0.0001
        )
        self.innerscheduler = lr_scheduler.StepLR(self.metainner_optim, step_size=1000, gamma=0.5)
        self.outerscheduler = lr_scheduler.StepLR(self.metaouter_optim, step_size=1000, gamma=0.5)
        now = datetime.now()
        self.date_time = now.strftime("%m-%d-%H-%M-%S")
        self.writer = SummaryWriter(os.path.join('./logs/log_{}'.format(self.date_time)))
        self.args=args
        self.init_meters()



    def train(self):
        training_process = []
        global best_model_wts
        best_acc = 0
        scheduler = lr_scheduler.StepLR(self.optimizers[0], step_size=50, gamma=0.5)
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])

            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)
            scheduler.step()


            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Val AUC:{val_result[0]:.2f}',
            ]))

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result )
            if best_acc < val_result[0] and epoch > 1:
                best_acc = val_result[0]
                best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model_wts)

        test_result = self.test_per_epoch(self.test_dataloader,
                                          self.test_loss, self.test_accuracy)
        self.logger.info(" | ".join([
            f'Test Loss:{self.test_loss.avg: .3f}',
            f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
            f'Test AUC:{test_result[0]:.2f}'
        ]))
        training_process.append(test_result)
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)

    def setup_path(self,repeat_time):

        self.root_folder = 'D:/learning/23/23.7/FBNETGEN-main/'
        self.datas = ['YALE1.h5',
                      'USM1.h5',
                      'UM1.h5',
                      'UCLA1.h5',
                      'TRINITY1.h5',
                      'STANFORD1.h5',
                      'SDSU1.h5',
                      'SBL1.h5',
                      'PITT1.h5',
                      'OLIN1.h5',
                      'NYU1.h5',
                      'MAX_MUN1.h5',
                      'LEUVEN1.h5',
                      'KKI1.h5',
                      'CMU1.h5',
                      'CALTECH1.h5']

        self.paths = []
        for data in self.datas:
            path = os.path.join(self.root_folder, data)
            self.paths.append(path)

        if repeat_time==0:
            self.unseen_index = [2]
        elif repeat_time==1:
            self.unseen_index = [3]
        else:
            self.unseen_index = [10]
        self.unseen_data_paths=[]

        for k in self.unseen_index:
            path=os.path.join(self.root_folder, self.datas[k])
            self.unseen_data_paths.append(path)
            self.paths.remove(self.paths[k])
            self.datas.remove(self.datas[k])
        print(self.unseen_data_paths)
        self.val_index=[0,0]
        self.batImageGenTrains = []
        self.val_paths = []
        for k in self.val_index:
            path=os.path.join(self.root_folder, self.datas[k])
            self.val_paths.append(path)
            self.paths.remove(self.paths[k])
            self.datas.remove(self.datas[k])

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(file_path=val_path, stage='val',
                                                 b_unfold_label=False)
            self.batImageGenVals.append(batImageGenVal)

        for train_path in self.paths:
            batImageGenTrain = BatchImageGenerator(file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)


    def sparsity_matrix_loss(self,matrix,label,len):
        mpos = torch.zeros(200, 200).to(device)

        mneg = torch.zeros(200, 200).to(device)
        labelpos=0
        labelneg=0
        for index,h in enumerate(label):
            if h==0:
                mpos=mpos+matrix[index]
                labelpos=labelpos+1
            else:
                mneg = mneg+matrix[index]
                labelneg = labelneg + 1
        if labelpos !=0:
            mpos = mpos / labelpos
        if labelneg !=0:
            mneg = mneg/labelneg

        overlap=abs(mpos-mneg)
        loss=torch.norm(overlap, p=1)/(200*200)
        return mpos,labelpos,mneg,labelneg,loss,overlap

    def overlap_loss(self,overlap1,overlap2):
        num=1
        overlap11=overlap1.view(num,-1)
        overlap22 = overlap1.view(num, -1)
        intersection = torch.sum(overlap11 * overlap22, dim=1)
        union = torch.sum(overlap11, dim=1) + torch.sum(overlap22, dim=1)
        dice_scores = 2 * intersection / (union + 1e-8)
        return 1 - dice_scores.mean()
    
    def matrix_entropy(self,matrix):

        log_matrix = torch.log(matrix + 1e-10)
        row_mean = torch.mean(log_matrix, dim=1)
        entropy = -torch.sum(torch.exp(row_mean) * row_mean, dim=0)
        return entropy

    def entropy_loss(self,matrix):
        probabilities = F.softmax(matrix, dim=0)
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=0)

        return entropy



    def train(self):
        self.model.train()
        self.best_accuracy_val = -1
        inner_loops =5000
        #self.model.load_state_dict(torch.load('models/'+'01-16-15-45-01'+'best.mdl'))



        for ite in range(inner_loops):

            self.model.train()
            meta_idx_all = np.random.choice(len(self.batImageGenTrains), 13, replace=False)
            meta_train_loss = 0.0
            for i in range(6):
                final_fc_train, final_pearson_train, labels_train = self.batImageGenTrains[
                    meta_idx_all[i]].get_images_labels_batch()
                final_fc_train, final_pearson_train, labels_train = torch.from_numpy(
                    np.array(final_fc_train, dtype=np.float32)), torch.from_numpy(
                    np.array(final_pearson_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))
                labels_train = labels_train.long()
                inputs_train, pearson_train, labels_train = final_fc_train.to(
                    device), final_pearson_train.to(device), labels_train.to(device)
                inputs, nodes, targets_a, targets_b, lam = mixup_data(
                    inputs_train, pearson_train, labels_train, 1, device)
                #outputs_train, _, _ = self.model(inputs_train, pearson_train)
                outputs_train, _, _ = self.model(inputs, nodes)
                torch.save(self.model.state_dict(), 'models/' + 'now.mdl')
                #loss = F.nll_loss(torch.log(outputs_train), labels_train)
                loss = 2 * mixup_criterion(
                    self.loss_fn, outputs_train, targets_a, targets_b, lam)
                meta_train_loss = meta_train_loss + loss

            self.metainner_optim.zero_grad()
            meta_train_loss.backward(retain_graph=True)
            self.metainner_optim.step()
            meta_idx_val=np.delete(meta_idx_all,[0,1,2,3,4,5])
            meta_val_loss=0.0
            overlap_all=[]
            outputs=[]
            labels=[]
            meta_val_loss_all=[]

            for k in meta_idx_val:
                batImageMetaVal=self.batImageGenTrains[k]
                final_fc_val, final_pearson_val, labels_val = batImageMetaVal.get_images_labels_batch()
                final_fc_val, final_pearson_val, labels_val = torch.from_numpy(
                    np.array(final_fc_val, dtype=np.float32)), torch.from_numpy(
                    np.array(final_pearson_val, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_val, dtype=np.float32))
                labels_val = labels_val.long()
                inputs_val, pearson_val, labels_val = final_fc_val.to(
                    device), final_pearson_val.to(device), labels_val.to(device)
                inputs, nodes, targets_a, targets_b, lam = mixup_data(
                    inputs_val, pearson_val, labels_val, 1, device)
                #outputs_val, learnable_matrix, edge_variance = self.model(inputs_val, pearson_val)
                outputs_val, learnable_matrix, edge_variance = self.model(inputs, nodes)
                outputs.append(outputs_val)
                labels.append(labels_val)
                croloss = 2 * mixup_criterion(
                    self.loss_fn, outputs_val, targets_a, targets_b, lam)

                meta_val_loss = meta_val_loss + croloss

                if self.sparsity_loss:
                    mpos, labelpos, mneg, labelneg, sparsity_loss, overlap = self.sparsity_matrix_loss(
                        learnable_matrix, labels_val, len(labels_val))


                    if labelpos != 0 and labelneg != 0:

                        #meta_val_loss = meta_val_loss + sparsity_loss * 1
                        mean = torch.mean(overlap, dim=0)
                        std = torch.std(overlap, dim=0)
                        overlap_normalized = (overlap - mean) / std
                        sigmoid_matrix = torch.sigmoid(overlap_normalized)


                        entropyloss=self.matrix_entropy(sigmoid_matrix)

                        meta_val_loss = meta_val_loss + entropyloss * 0.0001#sp
                        overlap_all.append(sigmoid_matrix)



            h=0
            for k in range(len(overlap_all)-1):
                for i in range(len(overlap_all)-k-1):
                    meta_val_loss = meta_val_loss + self.overlap_loss(overlap_all[i+k+1], overlap_all[k]) * 0.01#cons
                    h=h+1


            m = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6]], 0)
            label_val = torch.cat([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]], 0)
            acc = accuracy(m, label_val)[0]
            self.metaouter_optim.zero_grad()


            # backward your network
            meta_val_loss.backward()
            self.model.load_state_dict(torch.load('models/' + 'now.mdl'))
            self.model.train()
 
            self.metaouter_optim.step()
     

            flags_log = os.path.join('logs/', 'metalearning'+self.date_time+'read_log.txt')
            f = open(flags_log, mode='a')
            f.write(str(ite))
            f.write('train accuracy:')
            f.write(str(acc))
            f.close()
            val_acc=self.test_workflow(self.batImageGenVals,ite)
            self.writer.add_scalars('Acc', {'train_acc': acc,'val_acc':val_acc}, ite)
            f = open(flags_log, mode='a')
            f.write('val accuracy:')
            f.write(str(val_acc))
            f.write('\n')
            f.close()





    def write_log(self,log1, log2,log_path):
        f = open(log_path, mode='a')
        f.write('ite:')
        f.write(str(log1))
        f.write('accuracy:')
        f.write(str(log2))
        f.write('\n')
        f.close()

    def write_log2(self,log,log_path):
        f = open(log_path, mode='a')
        f.write('all_accuracy:')
        f.write(str(log))
        f.write('\n')
        f.close()

    def write_log3(self,log,log_path):
        f = open(log_path, mode='a')
        f.write('all_auc:')
        f.write(str(log))
        f.write('\n')
        f.close()


    def heldout_test(self):

        #self.model.load_state_dict(torch.load('models/' + '01-27-21-41-28best.mdl'))
        self.model.load_state_dict(torch.load('models/' + '01-28-20-12-37best.mdl'))
        self.batImageGenTests=[]
        self.model.eval()
        localtime = time.localtime(time.time())
        timestring = time.strftime('%Y/%m/%d - %H:%M:%S')
        flags_log = os.path.join('logs/', 'test_log.txt')
        #flags_log = os.path.join('logs/', 'con_loss_0.01_sp_0.0001_test_log.txt')
        self.write_log2(timestring, flags_log)
        for unseen_data_path in self.unseen_data_paths:
            batImageGenTest = BatchImageGenerator( file_path=unseen_data_path, stage='test',
                                                 b_unfold_label=False)
            self.batImageGenTests.append(batImageGenTest)

        accuracies = []
        results = []
        labels = []
        matrixs=[]
        pearson_matrixs=[]
        lengths = 0
        all_accuracies = 0
        for count, batImageGenTest in enumerate(self.batImageGenTests):
            accuracy_val,result,label,m,pearson_matrix= self.testout(batImageGenTest=batImageGenTest)
            print('ite:',count, 'accuracy:',accuracy_val)
            self.write_log(count,accuracy_val, flags_log)
            accuracies.append(accuracy_val)
            length = len(batImageGenTest.labels)
            all_accuracies = accuracy_val * length + all_accuracies
            lengths = length + lengths
            matrixs.extend(m)
            results.extend(result)
            labels.extend(label)
            pearson_matrixs.extend(pearson_matrix)
            #print(2)

        labels=np.array(labels)
        results = np.array(results)
        matrixs=np.array(matrixs)
        pearson_matrixs=np.array(pearson_matrixs)


        mean_acc = all_accuracies / lengths
        auc = roc_auc_score(labels, results)
        results = np.array(results)
        results[results > 0.5] = 1
        results[results <= 0.5] = 0
        precision, recall, f1_score, support = precision_recall_fscore_support(
            labels, results, average=None)
        f = open(flags_log, mode='a')
        f.write('precision:')
        f.write(str(precision))
        f.write('\n')
        f.write('recall:')
        f.write(str(recall))
        f.write('\n')
        f.write('f1_score:')
        f.write(str(f1_score))
        f.write('\n')
        f.write('support:')
        f.write(str(support))
        f.write('\n')
        f.close()
        self.write_log3(auc, flags_log)





    def testout(self, batImageGenTest):
        final_fc_test = batImageGenTest.final_fc
        final_pearson_test = batImageGenTest.final_pearson
        labels_test = batImageGenTest.labels
        average_matrix = np.mean(final_pearson_test, axis=0)
        threshold = 10
        result=[]
        labels=[]
        if len(labels_test) > threshold:
            n_slices_test = len(labels_test) // threshold
            indices_test = []

            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(labels_test) * (per_slice + 1) // n_slices_test)
            test_final_fc_splits = np.split(final_fc_test, indices_or_sections=indices_test)
            test_final_pearson_splits = np.split(final_pearson_test, indices_or_sections=indices_test)

            test_final_fc_splits_2_whole = np.concatenate(test_final_fc_splits)
            test_final_pearson_splits_2_whole = np.concatenate(test_final_pearson_splits)
            assert np.all(final_fc_test == test_final_fc_splits_2_whole), np.all(
                final_pearson_test == test_final_pearson_splits_2_whole)
            labels_preds = []
            matrix=[]
            pearson_matrix=[]
            for test_final_fc_split, test_final_pearson_split in zip(test_final_fc_splits,
                                                                     test_final_pearson_splits):
                final_fc_test, final_pearson_test = torch.from_numpy(
                    np.array(test_final_fc_split, dtype=np.float32)), torch.from_numpy(
                    np.array(test_final_pearson_split, dtype=np.float32))
                inputs_test, pearson_test = final_fc_test.to(
                    device), final_pearson_test.to(device)
                output, m, _ = self.model(inputs_test, pearson_test)

                output = output.cpu().data.numpy()
                m=m.cpu().data.numpy()
                pearson_test=pearson_test.cpu().data.numpy()
                pearson_matrix.append(pearson_test)
                matrix.append(m)
                labels_preds.append(output)
            matrixs= np.concatenate(matrix)
            pearson_matrix=np.concatenate(pearson_matrix)
            predictions = np.concatenate(labels_preds)
            labels_test = torch.from_numpy(
                np.array(labels_test, dtype=np.float32))
            labels_test = labels_test.long()
            predictions = torch.from_numpy(predictions)
            top1 = accuracy(predictions, labels_test)[0]
            #result +=predictions[:, 1].tolist()
            result += F.softmax(predictions, dim=1)[:, 1].tolist()
            labels += labels_test.tolist()
            return top1,result,labels,matrixs,pearson_matrix


    def test_workflow(self, batImageGenVals,ite):

        accuracies = []
        results=[]
        labels=[]
        predictions=[]

        lengths=0
        all_accuracies=0
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val,result,label,prediction = self.test2(batImageGenTest=batImageGenVal)
            accuracies.append(accuracy_val)
            results.extend(result)
            labels.extend(label)
            predictions.extend(prediction)
            length=len(batImageGenVal.labels)
            all_accuracies=accuracy_val*length+all_accuracies
            lengths=length+lengths
        flags_log = os.path.join('logs/', 'metalearning' + self.date_time + 'val_log.txt')
        f = open(flags_log, mode='a')
        f.write('\n')
        h=[0,10,20,30,40,50]
        for k in h:
            f.write('output:')
            f.write(str(predictions[k]))
            f.write(' ')
            f.write('result:')
            f.write(str(results[k]))
            f.write(' ')
            f.write('label:')
            f.write(str(labels[k]))
            f.write('\n')

        f.close()
        mean_acc=all_accuracies/lengths

        #auc = roc_auc_score(labels, results)


        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc
            torch.save(self.model.state_dict(),'models/'+self.date_time+'best.mdl')
        return mean_acc


    def test2(self,batImageGenTest):
        # switch on the network test mode
        labels = []
        result = []
        self.model.eval()
        final_fc_test=batImageGenTest.final_fc
        final_pearson_test=batImageGenTest.final_pearson
        labels_test = batImageGenTest.labels
        threshold = 10
        if len(labels_test) > threshold:
            n_slices_test = len(labels_test) // threshold
            indices_test = []

            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(labels_test) * (per_slice + 1) // n_slices_test)
            test_final_fc_splits = np.split(final_fc_test, indices_or_sections=indices_test)
            test_final_pearson_splits = np.split(final_pearson_test, indices_or_sections=indices_test)

            test_final_fc_splits_2_whole = np.concatenate(test_final_fc_splits)
            test_final_pearson_splits_2_whole = np.concatenate(test_final_pearson_splits)
            assert np.all(final_fc_test == test_final_fc_splits_2_whole),np.all(final_pearson_test == test_final_pearson_splits_2_whole)
            labels_preds = []
            for test_final_fc_split,test_final_pearson_split in zip(test_final_fc_splits,test_final_pearson_splits):
                final_fc_test, final_pearson_test = torch.from_numpy(
                    np.array(test_final_fc_split, dtype=np.float32)), torch.from_numpy(
                    np.array(test_final_pearson_split, dtype=np.float32))
                inputs_test, pearson_test = final_fc_test.to(
                    device), final_pearson_test.to(device)
                output, _, _ = self.model(inputs_test, pearson_test)
                output=output.cpu().data.numpy()
                labels_preds.append(output)
            predictions = np.concatenate(labels_preds)


            labels_test=torch.from_numpy(
            np.array(labels_test, dtype=np.float32))
            labels_test = labels_test.long()
            predictions=torch.from_numpy(predictions)
            top1 = accuracy(predictions, labels_test)[0]
            result += F.softmax(predictions, dim=1)[:, 1].tolist()
            labels += labels_test.tolist()

            return top1,result,labels,predictions


