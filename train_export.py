"""Train a 3->8->3 quantization-aware MLP and export Nano PROGMEM weights.
Usage: python train_export.py dataset.csv
CSV columns: distance_cm,closing_cm_s,delta_cm,label
label must be SAFE, CAUTION, or DANGER. No synthetic data is supplied.
"""
import csv, sys, pathlib
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

LABELS=['SAFE','CAUTION','DANGER']
BASE=pathlib.Path(__file__).resolve().parent

def quant(w):
    # Q4 signed int8 fixed scale 1/16, clamp and straight-through estimator
    v=torch.clamp(torch.round(w*16),-128,127)/16
    return w+(v-w).detach()

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(3,8)
        self.fc2=nn.Linear(8,3)
    def forward(self,x):
        h=torch.relu(nn.functional.linear(x,quant(self.fc1.weight),quant(self.fc1.bias)))
        h=torch.clamp(h,0,1)
        h=h+(torch.round(h*127)/127-h).detach()
        return nn.functional.linear(h,quant(self.fc2.weight),quant(self.fc2.bias))

def integer_predict(X,W1,B1,W2,B2):
    x=np.clip(np.floor(X*np.array([127/200,127/100,127/50])+0.5),0,127).astype(np.int32)
    acc=x@W1.T+127*B1
    h=np.clip(np.trunc(acc/16),0,127).astype(np.int32)
    logits=h@W2.T+127*B2
    return logits.argmax(axis=1)

def main(path):
    rows=list(csv.DictReader(open(path,newline='',encoding='utf-8-sig')))
    if len(rows)<100:
        raise SystemExit('Need at least 100 labelled real samples; 1000+ across separate setups recommended.')
    X=np.array([[float(r['distance_cm']),float(r['closing_cm_s']),float(r['delta_cm'])] for r in rows],dtype=np.float32)
    y=np.array([LABELS.index(r['label'].strip().upper()) for r in rows])
    if len(set(y))!=3: raise SystemExit('All three labels SAFE, CAUTION, DANGER are required.')
    # Use separate sessions/scenes for validation if possible; random split is a preliminary check only.
    X[:,0]=np.clip(X[:,0],0,200);X[:,1]=np.clip(X[:,1],0,100);X[:,2]=np.clip(X[:,2],0,50)
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    torch.manual_seed(42)
    net=TinyNet()
    optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    xt=torch.tensor(Xtrain/np.array([200,100,50],dtype=np.float32))
    yt=torch.tensor(ytrain,dtype=torch.long)
    for epoch in range(350):
        net.train();optimizer.zero_grad()
        loss=nn.functional.cross_entropy(net(xt),yt)
        loss.backward();optimizer.step()
    def arr(param): return np.clip(np.rint(param.detach().numpy()*16),-128,127).astype(np.int32)
    W1=arr(net.fc1.weight);B1=arr(net.fc1.bias)
    W2=arr(net.fc2.weight);B2=arr(net.fc2.bias)
    pred=integer_predict(Xtest,W1,B1,W2,B2)
    print('Integer inference evaluation (random split; also test unseen physical scenes):')
    print(classification_report(ytest,pred,target_names=LABELS,zero_division=0))
    print('Confusion matrix:\n',confusion_matrix(ytest,pred))
    def matrix(a):return '{'+',\n'.join(' {'+','.join(map(str,row))+'}' for row in a)+'\n}'
    def vector(a):return '{'+','.join(map(str,a))+'}'
    header='''#pragma once
#include <Arduino.h>
#include <avr/pgmspace.h>
// Generated from measured and labelled training data. Q4 weights/biases: real=int8/16.
#define MODEL_TRAINED 1
'''
    header+=f'const int8_t W1[8][3] PROGMEM = {matrix(W1)};\n'
    header+=f'const int8_t B1[8] PROGMEM = {vector(B1)};\n'
    header+=f'const int8_t W2[3][8] PROGMEM = {matrix(W2)};\n'
    header+=f'const int8_t B2[3] PROGMEM = {vector(B2)};\n'
    output=BASE/'SONARQ_Nano'/'model_weights.h'
    output.write_text(header)
    print('Exported:',output)

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python train_export.py dataset.csv')
    main(sys.argv[1])
