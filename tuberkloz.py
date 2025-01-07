import os #--------------------------------------------------------------
import numpy as np #--------------------------------------------------------------
import matplotlib.pyplot as plt #--------------------------------------------------------------
import tensorflow as tf #--------------------------------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator #--------------------------------------------------------------
from tensorflow.keras.layers import Input #--------------------------------------------------------------
from tensorflow.keras import layers, models #--------------------------------------------------------------
from sklearn.metrics import classification_report #--------------------------------------------------------------
#--------------------------------------------------------------
data_dir = '/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'
data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.1) #--------------------------------------------------------------
train_data = data_gen.flow_from_directory(data_dir, target_size=(150, 150), batch_size=32, class_mode='binary',subset='training')
val_data = data_gen.flow_from_directory( data_dir,target_size=(150, 150),batch_size=32,class_mode='binary',subset='validation')
model = models.Sequential([Input(shape=(150, 150, 3)),layers.Conv2D(32, (3, 3), activation='relu'),layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),layers.MaxPooling2D((2, 2)),layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),layers.Flatten(), layers.Dense(128, activation='relu'),layers.Dense(1, activation='sigmoid')])#----------
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#----------
history = model.fit(train_data,epochs=10,validation_data=val_data)#------------
#--------------------------------------------------------------
plt.figure()#--------------------------------------------------------------
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')#--------------------------------------------------------------
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')#--------------------------------------------------------------
plt.xlabel('epoklar')#--------------------------------------------------------------
plt.ylabel('Doğruluk değerleri')#--------------------------------------------------------------
plt.legend()#--------------------------------------------------------------
plt.title('Doğruluk Grafiği ')#--------------------------------------------------------------
plt.show()#--------------------------------------------------------------
plt.figure()#--------------------------------------------------------------
plt.plot(history.history['loss'], label='eğitimde doğruluk oranı')#--------------------------------------------------------------
plt.plot(history.history['val_loss'], label='eğitimde kayıp oranları')#--------------------------------------------------------------
plt.xlabel('Epochs')#--------------------------------------------------------------
plt.ylabel('Kayıp')#--------------------------------------------------------------
plt.legend()#--------------------------------------------------------------
plt.title('Kayıp Grafiği')#--------------------------------------------------------------
plt.show()#--------------------------------------------------------------
class_names = ['Normal', 'Tuberculosis']#--------------------------------------------------------------
plt.figure(figsize=(10, 10))#--------------------------------------------------------------
for i in range(5):#--------------------------------------------------------------
    img_batch, label_batch = next(val_data)  #--------------------------------------------------------------
    img = img_batch[0] #--------------------------------------------------------------
    label = label_batch[0]  #--------------------------------------------------------------
    prediction = model.predict(np.expand_dims(img, axis=0))#--------------------------------------------------------------
    plt.subplot(1, 5, i + 1)#--------------------------------------------------------------
    plt.imshow((img * 255).astype("uint8"))  #--------------------------------------------------------------
    plt.title(f"Gerçek değeri: {class_names[int(label)]}\n Makinenin Tahmini değeri: {class_names[int(prediction[0] > 0.5)]}")#--------------------------------------------------------------
    plt.axis('off')#--------------------------------------------------------------
plt.show() #--------------------------------------------------------------
val_data.reset()  #--------------------------------------------------------------
y_true = val_data.classes#--------------------------------------------------------------#----------
y_pred = (model.predict(val_data) > 0.5).astype('int32')#--------------------------------------------------------------#----------
print(classification_report(y_true, y_pred, target_names=class_names))#--------------------------------------------------------------
#--------------------------------------------------------------#----------#----------
#--------------------------------------------------------------#----------#----------
#--------------------------------------------------------------#----------#----------
#--------------------------------------------------------------#----------#----------#----------#----------
#----------#----------#----------#----------#----------#----------#----------#----------#----------#----------#----------#----------



