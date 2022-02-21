import numpy as np
from skimage.feature import hog
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
features=[]
path=os.listdir('C:/Users/LTC/PycharmProjects/task4')
for i in path:
    read=cv2.imread(i)

    resiz=cv2.resize(read,(128,64))
    fd,hog_=hog(resiz,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
    features.append(fd)
    #name=i.split('.')
    
    #y.append( 1 if name[0]=='cat' else 0)

C = 0.1

xtrain,xtest,ytrain,ytest=train_test_split(features,y,test_size=0.3,random_state=0,shuffle=True)


rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(xtrain, ytrain)
rbf_svc=svm.SVC().fit(xtrain, ytrain)
prediction=svm.SVC().predict(xtest)


accuracy=np.mean(prediction==ytest)
print(accuracy)
