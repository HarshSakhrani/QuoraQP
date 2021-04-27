#Importing all the necessary libraries
from pycm import *
from transformers import BertTokenizer, BertModel
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import pickle
import sys
from glob import glob  
import math
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
import torch.utils.data.dataloader
import torchvision.transforms as visionTransforms
import PIL.Image as Image
from torchvision.transforms import ToTensor,ToPILImage

df=pd.read_csv("train.csv",index_col=0)

#10K
dfTrain,dfVal,dfTest=np.split(df.sample(frac=1, random_state=42), [int(.9505 * len(df)), int(.9752 * len(df))])
dfTrain=dfTrain.reset_index(drop=True)
dfTest=dfTest.reset_index(drop=True)
dfVal=dfVal.reset_index(drop=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import WeightedRandomSampler
freqLabels=torch.tensor(dfTrain['is_duplicate'].value_counts().sort_index(),dtype=torch.double)
weightClass=freqLabels/freqLabels.sum()
weightClass= 1/weightClass
weightClass=(weightClass).tolist()
sampleWeights=[weightClass[i] for i in dfTrain['is_duplicate']]
trainSampler=WeightedRandomSampler(sampleWeights,len(dfTrain))

from torch.utils.data import Dataset, DataLoader

class QuoraDataset(Dataset):

  def __init__(self,dataframe,bertTokenizer,maxLength,device):
    self.data=dataframe
    self.bertTokenizer=bertTokenizer
    self.maxLength=maxLength
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.q1=str(self.data.iloc[idx,2])
    self.q2=str(self.data.iloc[idx,3])
    self.label=self.data.iloc[idx,4]

    self.encodedStr=self.bertTokenizer.encode_plus(self.q1,self.q2,
						padding='max_length',
						truncation="longest_first",
						max_length=self.maxLength,
						return_tensors='pt',
						return_attention_mask=True,
						return_token_type_ids=True).to(device)

    return self.encodedStr,self.label

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
quoraTrainDataset=QuoraDataset(dataframe=dfTrain,bertTokenizer=tokenizer,maxLength=64,device=device)
quoraTestDataset=QuoraDataset(dataframe=dfTest,bertTokenizer=tokenizer,maxLength=64,device=device)
quoraValDataset=QuoraDataset(dataframe=dfVal,bertTokenizer=tokenizer,maxLength=64,device=device)

trainLoader=torch.utils.data.DataLoader(quoraTrainDataset,batch_size=8,sampler=trainSampler)
testLoader=torch.utils.data.DataLoader(quoraTestDataset,batch_size=8,shuffle=True)
valLoader=torch.utils.data.DataLoader(quoraValDataset,batch_size=8,shuffle=True)

class KimCNN(nn.Module):
  def __init__(self,preTrainedBert,inChannels=1,embeddingDimension=768,numClasses=2):
    super(KimCNN,self).__init__()

    self.inChannels=inChannels
    self.embDim=embeddingDimension
    self.numClasses=numClasses

    self.bert=self.freezeBert(preTrainedBert)
    
    self.kimConv0=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(2,self.embDim))
    self.kimConv1=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(3,self.embDim))
    self.kimConv2=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(4,self.embDim))
    self.kimConv3=nn.Conv2d(in_channels=self.inChannels,out_channels=100,kernel_size=(5,self.embDim))
    self.dropoutLayer=nn.Dropout(p=0.5)
    self.fc=nn.Linear(400,1)

  def forward(self,input):

    bertOutput=self.bert(input_ids=input['input_ids'].squeeze(dim=1),attention_mask=input['attention_mask'].squeeze(dim=1),token_type_ids=input['token_type_ids'].squeeze(dim=1)).last_hidden_state
    
    kimInput=bertOutput.unsqueeze(1)
    
    conv0_Output=F.relu(self.kimConv0(kimInput)).squeeze(3)
    conv1_Output=F.relu(self.kimConv1(kimInput)).squeeze(3)
    conv2_Output=F.relu(self.kimConv2(kimInput)).squeeze(3)
    conv3_Output=F.relu(self.kimConv3(kimInput)).squeeze(3)
    
    conv0_Output=F.max_pool1d(conv0_Output,conv0_Output.size(2))
    conv1_Output=F.max_pool1d(conv1_Output,conv1_Output.size(2))
    conv2_Output=F.max_pool1d(conv2_Output,conv2_Output.size(2))
    conv3_Output=F.max_pool1d(conv3_Output,conv3_Output.size(2))

    kimOutput=torch.cat((conv0_Output.squeeze(dim=2),conv1_Output.squeeze(dim=2),conv2_Output.squeeze(dim=2),conv3_Output.squeeze(dim=2)),dim=1)

    output=self.fc(self.dropoutLayer(kimOutput))

    output=output.reshape((output.size(0)))

    return output

  def freezeBert(self,model):
    count=0
    for name,params in model.named_parameters():
      count=count+1
      if count<134:
        params.requires_grad=False

    return model

model = BertModel.from_pretrained('bert-base-uncased')
kimcnn=KimCNN(preTrainedBert=model)
kimcnn.to(device)
bceLoss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(kimcnn.parameters(), lr=0.00001)

def Average(lst): 
    return sum(lst) / len(lst) 

def train_model(model,epochs):

  trainBatchCount=0
  testBatchCount=0

  avgTrainAcc=[]
  avgValidAcc=[]
  trainAcc=[]
  validAcc=[]
  trainLosses=[]
  validLosses=[]
  avgTrainLoss=[]
  avgValidLoss=[]


  for i in range(epochs):

    print("Epoch:",i)

    model.train()
    print("Training.....")
    for batch_idx,(input,targets) in enumerate(trainLoader):

      trainBatchCount=trainBatchCount+1

      targets=targets.to(device)
      
      optimizer.zero_grad()

      scores=model(input)

      targets=targets.type_as(scores)
       
      loss=bceLoss(scores,targets)

      loss.backward()

      optimizer.step()

      trainLosses.append(float(loss))
      
      correct=0
      total=0
      total=len(targets)

      predictions=torch.round(nn.Sigmoid()(scores))
      correct = (predictions==targets).sum()
      acc=float((correct/float(total))*100)

      trainAcc.append(acc)

      if ((trainBatchCount%200)==0):

        print("Targets:-",targets)
        print("Predictions:-",predictions)

        print('Loss: {}  Accuracy: {} %'.format(float(loss), acc))

    model.eval()
    print("Validating.....")
    for batch_idx,(input,targets) in enumerate(valLoader):

      testBatchCount=testBatchCount+1

      scores=model(input)
      targets=targets.to(device)
      targets=targets.type_as(scores)

      loss=bceLoss(scores,targets)

      validLosses.append(float(loss))

      testCorrect=0
      testTotal=0

      predictions=torch.round(nn.Sigmoid()(scores))

      testCorrect = (predictions==targets).sum()
      testTotal=predictions.size(0)

      testAcc=float((testCorrect/float(testTotal))*100)

      validAcc.append(testAcc)

      if ((testBatchCount%200)==0):
        print('Loss: {}  Accuracy: {} %'.format(float(loss), testAcc))
    

    trainLoss=Average(trainLosses)
    validLoss=Average(validLosses)
    avgTrainLoss.append(trainLoss)
    avgValidLoss.append(validLoss)
    tempTrainAcc=Average(trainAcc)
    tempTestAcc=Average(validAcc)
    avgTrainAcc.append(tempTrainAcc)
    avgValidAcc.append(tempTestAcc)

    print("Epoch Number:-",i,"  ","Training Loss:-"," ",trainLoss,"Validation Loss:-"," ",validLoss,"Training Acc:-"," ",tempTrainAcc,"Validation Acc:-"," ",tempTestAcc)

    trainAcc=[]
    ValidAcc=[]
    trainLosses=[]
    validLosses=[]

  return model,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc


kimcnn,avgTrainLoss,avgValidLoss,avgTrainAcc,avgValidAcc=train_model(kimcnn,12)

loss_train = avgTrainLoss
loss_val = avgValidLoss
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss(KimCNN_BCELoss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png', bbox_inches='tight')
plt.close()
#plt.show()

loss_train = avgTrainAcc
loss_val = avgValidAcc
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training Acc')
plt.plot(epochs, loss_val, 'b', label='Validation Acc')
plt.title('Training and Validation loss(KimCNN_BCELoss)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Acc.png', bbox_inches='tight')
plt.close()

def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for input,targets in loader:

      targets=targets.to(device=device)

      scores=model(input)
      targets=targets.type_as(scores)
      predictions=torch.round(nn.Sigmoid()(scores))


      targets=targets.tolist()
      predictions=predictions.tolist()


      completeTargets.append(targets)
      completePreds.append(predictions)

    completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return cm

cm=checkClassificationMetrics(testLoader,kimcnn)

f=open("Results.txt","a")
f.write(str(cm))
f.close()

