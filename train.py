import pandas as pd
import numpy as np
import fasttext

traindata = pd.read_csv("/content/drive/MyDrive/BBang/2. category/train.csv",encoding='cp949').astype(str)
testdata = pd.read_csv("/content/drive/MyDrive/BBang/2. category/test.csv", sep = ',',encoding='cp949').astype(str)
print(traindata)

df2 = pd.DataFrame(columns=['category','document'])
df2['category'] = '__label__'+traindata['category']
df2['document'] = traindata['document']
# print(df2)

df2.to_csv('labelingtrain.txt', sep = '\t', index = False)
labeling = pd.read_csv("/content/labelingtrain.txt", sep = '\t')

model = fasttext.train_supervised('/content/labelingtrain.txt', wordNgrams=3, epoch=25, lr=0.35)

predictions=[]
for line in testdata['document']:
  pred_label=model.predict(line, threshold=0.5)[0]
  predictions.append(pred_label)

