
import TextClassifier 
import numpy as np
import csv
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from visdom import Visdom
import time
import random

def data_import(mode):
    cu_path = os.getcwd()
    name = mode
    train_path_X = os.path.join(cu_path,"data",name+"_X.csv")
    train_path_Y = os.path.join(cu_path,"data",name+"_Y.csv")

    sent_vector = np.loadtxt(train_path_X, delimiter=',')
    gold_label_vector = np.loadtxt(train_path_Y)
    gold_label_vector = torch.from_numpy(gold_label_vector).long()

    if torch.cuda.is_available():
        gold_label_vector = gold_label_vector.cuda()

        #to deviceとどっちがいいのか
        


    return sent_vector,gold_label_vector



if __name__ == '__main__':

    #次元数を指定 
    embedding_dim = 300
    label_num = 4
    # print(embedding_dim) #動作確認
    # print(label_num) #動作確認

    #model 呼び出し

    #problem 77~79
    viz = Visdom()
    epoch_num = 200
    batch_size = 32
    # batch_list = [2,4,8,16,32,64]

    batch_list = [64]

    for batch_size in batch_list:
        #みにバッチをto.deviceでGPUに送るといい
        #普通は全部は乗らない
        
        print("batch_num",batch_size)

        model = TextClassifier.TextClassification(embedding_dim,label_num)
        train_X, train_y = data_import("train")
        dev_X, dev_y = data_import("dev")

        #detaset DataSetを使おう！！
        #舟山のやつを見よう
        #Tensor_data set を使うとx,yをパッキングできる

        train_num = len(train_X)

        if train_num % batch_size == 0:
            total_batch = train_num // batch_size 
        else:
            total_batch = train_num // batch_size + 1

        train_loss_pool = []
        dev_loss_pool = []
        train_loss = 0
        dev_loss = 0

        train_acc_pool= []
        dev_acc_pool = []
        train_acc = 0
        dev_acc = 0

        prev_acc = -1

        model.zero_grad()
        optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
        optimizer.zero_grad()

        for epoch in range(epoch_num):
            if epoch == epoch_num-1:
                temp_start = time.time()
            
            # train data 
            model.train()
            sample_loss = 0

            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size

                instance_X = train_X[start:end]
                instance_y = train_y[start:end]


                pred_y = model(instance_X)
                pred_y = torch.log(pred_y)
                
                # print(pred_y)
                # for batch_iter 
                loss = F.nll_loss(pred_y,instance_y)
                loss.backward()
                sample_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            if epoch == epoch_num-1: 
                temp_end = time.time()
                time_cost = temp_end - temp_start
                print(time_cost)

            pred_y = model(train_X)
            pred_y = torch.log(pred_y)
            pred_y = torch.argmax(pred_y,dim=1)
            
            if torch.cuda.is_available():
                train_acc = accuracy_score(train_y.cpu(), pred_y.cpu())
            else:
                train_acc = accuracy_score(train_y, pred_y)
            train_acc_pool.append(train_acc)
            train_loss += sample_loss/total_batch
            train_loss_pool.append(train_loss)


            #dev data
            model.eval()
            pred_y = model(dev_X)
            pred_y = torch.log(pred_y)
            loss = F.nll_loss(pred_y,dev_y)
            dev_loss += loss.item()
            dev_loss_pool.append(dev_loss)


            pred_y = torch.argmax(pred_y,dim=1)
            if torch.cuda.is_available():
                dev_acc = accuracy_score(dev_y.cpu(), pred_y.cpu())
            else:
                dev_acc = accuracy_score(dev_y, pred_y)
            dev_acc_pool.append(dev_acc)

            if dev_acc > prev_acc :
                prev_acc = dev_acc
                model_name = "work/model/"+ str(epoch) + ".model"
                torch.save(model.state_dict(), model_name)

                optim_name = "work/model/"+str(epoch)+".opt"
                torch.save(optimizer.state_dict(), optim_name)


            
            if not torch.cuda.is_available():
                viz.line(X=np.array([epoch]), Y=np.array([train_loss/(epoch+1)]), win='loss', \
                name='train_loss', update='append')

                viz.line(X=np.array([epoch]), Y=np.array([dev_loss/(epoch+1)]), win='loss', \
                name='dev_loss', update='append')

                viz.line(X=np.array([epoch]), Y=np.array([train_acc]), win='acc', \
                name='train_acc', update='append')

                viz.line(X=np.array([epoch]), Y=np.array([dev_acc]), win='acc', \
                name='dev_acc', update='append')

        #test
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_name))
        else:
            model.load_state_dict(torch.load(model_name,map_location='cpu'))

        test_X, test_y = data_import("test")
        pred_y = model(test_X)
        pred_y = torch.log(pred_y)
        pred_y = torch.argmax(pred_y,dim=1)

        if torch.cuda.is_available():
            test_acc = accuracy_score(test_y.cpu(), pred_y.cpu())
        else:
            test_acc = accuracy_score(test_y, pred_y)
            
        print("final test score is ",test_acc)

    """
    sent_vector,gold_label_vector= data_import("train")

    model = TextClassifier.TextClassification(embedding_dim,label_num)
    pred_label_vector = model(sent_vector)
    pred_label_vector = torch.log(pred_label_vector)

    #problem71
    y_1 = model(sent_vector[0])
    print("y_1\n",y_1)
    Y = model(sent_vector[:4])
    print("Y\n",Y)
    

    # problem 72
    # nll_lossはone-hotではなくクラス番号をtargetにとるので注意
    loss = F.nll_loss(pred_label_vector[:4],gold_label_vector[:4],reduction='mean')
    loss.backward()

    print("損失\n",loss.item())
    print("重み行列\n",model.W.grad)


    #problem73
    train_X, train_y = data_import("train")
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9)
    optimizer.zero_grad()

    for epoch in range(100):
        pred_y = model(train_X)
        loss = F.nll_loss(pred_y,train_y,reduction='mean')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(loss)


    #problem74 
    with torch.no_grad():
        for mode in ["train","test"]:
            X,gold_y = data_import(mode)
            pred_y = model(X)
            pred_y = torch.argmax(pred_y,dim=1)
            print("mode is",mode)
            print(accuracy_score(gold_y, pred_y))



    #problem75 ~ 76
    #可視化ツールらしい
    viz = Visdom()
    epoch_num = 1000

    train_X, train_y = data_import("train")
    test_X, test_y = data_import("dev")
    train_loss_pool = []
    test_loss_pool = []
    train_loss = 0
    test_loss = 0

    train_acc_pool= []
    test_acc_pool = []
    train_acc = 0
    test_acc = 0

    model.zero_grad()
    optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9)
    optimizer.zero_grad()

    for epoch in range(epoch_num):
        
        # train data 
        model.train()
        pred_y = model(train_X)
        pred_y = torch.log(pred_y)
        loss = F.nll_loss(pred_y,train_y)
        loss.backward()
        train_loss += loss.item()
        train_loss_pool.append(train_loss)

        pred_y = torch.argmax(pred_y,dim=1)
        train_acc = accuracy_score(train_y, pred_y)
        train_acc_pool.append(train_acc)

        optimizer.step()


        #test data
        model.eval()
        pred_y = model(test_X)
        pred_y = torch.log(pred_y)
        loss = F.nll_loss(pred_y,test_y)
        test_loss += loss.item()
        test_loss_pool.append(test_loss)


        pred_y = torch.argmax(pred_y,dim=1)
        test_acc = accuracy_score(test_y, pred_y)
        test_acc_pool.append(test_acc)

        

        model_name = "work/model/"+ str(epoch_num) + ".model"
        torch.save(model.state_dict(), model_name)

        optim_name = "work/model/"+str(epoch_num)+".opt"
        torch.save(optimizer.state_dict(), optim_name)
        


        viz.line(X=np.array([epoch]), Y=np.array([train_loss/(epoch+1)]), win='loss', \
        name='train_loss', update='append')

        viz.line(X=np.array([epoch]), Y=np.array([test_loss/(epoch+1)]), win='loss', \
        name='test_loss', update='append')

        viz.line(X=np.array([epoch]), Y=np.array([train_acc]), win='acc', \
        name='train_acc', update='append')

        viz.line(X=np.array([epoch]), Y=np.array([test_acc]), win='acc', \
        name='test_acc', update='append')

        optimizer.zero_grad()
        """


        













    







