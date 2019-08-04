import cv2
import numpy as np
import os
import tkinter
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
lefteye_cascade=cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
righteye_cascade=cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
train_images=[]
train_labels=[]
test_images=[]
test_labels=[]
tn_images=[]
ts_images=[]



def train(d_name,label):
    for filename in os.listdir(d_name):
        img=cv2.imread(os.path.join(d_name,filename))
        if img is not None:
             img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             eyes=lefteye_cascade.detectMultiScale(img,1.1,3)
             for (x,y,w,h) in eyes:
                 imgt=img[y:y+h, x:x+w]
                 h, w=imgt.shape[:2]
                 if((h!=0)&(w!=0)):
                     #cv2.imshow('',imgt)
                     #cv2.waitKey(400)
                     train_images.append(imgt)
                     train_labels.append(label)
                 break
             eyes=righteye_cascade.detectMultiScale(img,1.1,3)
             for (x,y,w,h) in eyes:
                 img=img[y:y+h, x:x+w]
                 h, w=img.shape[:2]
                 if((h!=0)&(w!=0)):
                     #cv2.imshow('',img)
                     #cv2.waitKey(400)
                     train_images.append(img)
                     train_labels.append(label)
                 break

def test(d_name,label):
    for filename in os.listdir(d_name):
        img=cv2.imread(os.path.join(d_name,filename))
        if img is not None:
             img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             eyes=lefteye_cascade.detectMultiScale(img,1.1,3)
             for (x,y,w,h) in eyes:
                 imgt=img[y:y+h, x:x+w]
                 h, w=imgt.shape[:2]
                 if((h!=0)&(w!=0)):
                     #cv2.imshow('',imgt)
                     #cv2.waitKey(400)
                     test_images.append(imgt)
                     test_labels.append(label)
                 break
             eyes=righteye_cascade.detectMultiScale(img,1.1,3)
             for (x,y,w,h) in eyes:
                 img=img[y:y+h, x:x+w]
                 h, w=img.shape[:2]
                 if((h!=0)&(w!=0)):
                     #cv2.imshow('',img)
                     #cv2.waitKey(400)
                     test_images.append(img)
                     test_labels.append(label)
                 break



            
s=30
#train('angry','sad')
train('open','open')
train('close','close')
#test('a','sad')
test('h','open')
test('s','close')
cv2.destroyAllWindows()
for img in train_images:
    img=cv2.resize(img,(s,s))
    #cv2.imshow('',img)
    #cv2.waitKey(300)
    img=img.flatten()
    tn_images.append(img)
cv2.destroyAllWindows()

print(len(tn_images))
print(len(train_labels))

for img in test_images:
    img=cv2.resize(img,(s,s))
    #cv2.imshow('',img)
    #cv2.waitKey(300)
    img=img.flatten()
    ts_images.append(img)
print('test')

cv2.destroyAllWindows()

tn_images=np.array(tn_images)
ts_images=np.array(ts_images)
train_labels=np.array(train_labels)
test_labels=np.array(test_labels)
#(xtrain, xtest, ytrain, ytest) = train_test_split(tn_images, train_labels, test_size=0.1)
#for k in range (1,12):
model = KNeighborsClassifier(n_neighbors=4)
model.fit(tn_images,train_labels)
#model.fit(xtrain, ytrain)
acc = model.score(ts_images,test_labels)
#acc = model.score(xtest, ytest)
print("accuracy: {:.2f}%".format(acc * 100))

def openfile():
    os.startfile("d:\\selfie_eye\\smile")

def selfie():
    cam= cv2.VideoCapture(0)
    i=0
    flag=0
    ret=True
    while(ret):    
        ret, image=cam.read()
        if(ret==True):
            eye=[]
            img=cv2.resize(image,(550,550)) 
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            eyes=lefteye_cascade.detectMultiScale(gray,1.1,3)
            k=0
            for (x,y,w,h) in eyes:
                imgt=gray[y:y+h, x:x+w]
                h, w=imgt.shape[:2]
                if((h!=0)&(w!=0)):
                    img_le=gray[y:y+h, x:x+w]
                    img_le=cv2.resize(img_le,(s,s)).flatten()
                    img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    #img_face=np.array(img_face)
                    eye.append(img_le)
                    k=k+1
                    break
    
                    
            eyes=righteye_cascade.detectMultiScale(gray,1.1,3)
            for (x,y,w,h) in eyes:
                imgt=gray[y:y+h, x:x+w]
                h, w=imgt.shape[:2]
                if((h!=0)&(w!=0)):
                    img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    img_re=gray[y:y+h, x:x+w]
                    img_re=cv2.resize(img_re,(s,s)).flatten()
                    
                       #img_face=np.array(img_face)
                    eye.append(img_re)
                    k=k+1
                    break
            eye=np.array(eye)
            value1=''
            value2=''
                #img_face=img_face.reshape(1,-1)
            if(k>0):
                pred=model.predict(eye)
                value1=pred[0]
            if(k==2):
                value2=pred[1]
                #img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            print(value1)
            print("\n")
            print(value2)
            if(value1=='close' or value2=='close'):
                i=i+1
                print("\n")
                flag=flag+1
                print(i)
            else:
                #print("no\n")
                flag=0
                    
                #cv2.putText(img,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
                #break
            
            if(cv2.waitKey(30) == ord('q')):
                print("interrupted")
                break
            if(flag==10):
                smile='smile/sm0'+str(i)+'.jpg'
                cv2.imwrite(smile,image)
                break
            cv2.imshow("EMOTION DETECTOR",img)

    cam.release()
    cv2.destroyAllWindows()

root=tkinter.Tk()
root.title("Smart Selfie")
label=tkinter.Label(root,text="Smart Selfie : Don't forget to Smile")
label.pack()
selfie=tkinter.Button(root,text="Take a Selfie",command=selfie)
selfie.pack()
browser=tkinter.Button(root,text="Open Selfie Folder",command=openfile)
browser.pack()
root.mainloop()


 
