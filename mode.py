from keras.models import Sequential
from keras.constraints import *
from keras.optimizers import *
from keras.utils import np_utils
from keras import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator


#original size (696, 520)
f=256
s=256
#first model
main_model = Sequential()
main_model.add(Conv2D(32, kernel_size=3, input_shape=(f, s, 1),activation='relu'))
main_model.add(BatchNormalization())
main_model.add(MaxPool2D(strides=(5,5)))
main_model.add(Dropout(0.5))
main_model.add(Conv2D(32, kernel_size=3,activation='relu'))
main_model.add(BatchNormalization())
main_model.add(MaxPool2D(strides=(5,5)))
main_model.add(Dropout(0.5))
main_model.add(Conv2D(64, kernel_size=3,activation='relu'))
main_model.add(BatchNormalization())
main_model.add(MaxPool2D(strides=(5,5)))
main_model.add(Dropout(0.5))
#main_model.add(Conv2D(64, kernel_size=3,activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(MaxPool2D(strides=(5,5)))
#main_model.add(Dropout(0.5))
main_model.add(Flatten())

#lower features model - CNN2
lower_model1 = Sequential()
lower_model1.add(MaxPool2D(strides=(5,5), input_shape=(f, s,1)))
lower_model1.add(Conv2D(32, kernel_size=3,activation='relu'))
lower_model1.add(BatchNormalization())
lower_model1.add(MaxPool2D(strides=(5,5)))
lower_model1.add(Dropout(0.5))
lower_model1.add(Conv2D(32, kernel_size=3,activation='relu'))
lower_model1.add(BatchNormalization())
lower_model1.add(MaxPool2D(strides=(5,5)))
lower_model1.add(Dropout(0.5))
#lower_model1.add(Conv2D(64, kernel_size=3,activation='relu'))
#lower_model1.add(BatchNormalization())
#lower_model1.add(MaxPool2D(strides=(5,5)))
#lower_model1.add(Dropout(0.5))
lower_model1.add(Flatten())

#merged model
merged_model = Concatenate()([main_model.output, lower_model1.output])

x = Dense(128, activation='relu')(merged_model)
x = Dropout(0.25)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(32, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

# add in dense layer activity_regularizer=regularizers.l1(0.01)

final_model = Model(inputs=[main_model.input, lower_model1.input], outputs=[output])

final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



traindir1="/content/drive/My Drive/DL_2_Dataset/bbrod/train"
traindir2="/content/drive/My Drive/DL_2_Dataset/bbrod/train"

testdir1="/content/drive/My Drive/DL_2_Dataset/bbrod/train"
testdir2="/content/drive/My Drive/DL_2_Dataset/bbrod/train"


input_imgen = ImageDataGenerator(rescale = 1./255,
   rotation_range=80,
    width_shift_range=0.6,
  height_shift_range=0.5,
    horizontal_flip=True,zoom_range=0.8,vertical_flip=True,
    validation_split=0.4)

test_imgen = ImageDataGenerator(rescale = 1./255)

batch_size=16

def generate_generator_multiple(generator,dir1, dir2, batch_size, img_height,img_width,subset):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False, color_mode='grayscale',
                                          seed=7,subset=subset)
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False, color_mode='grayscale',
                                          seed=7,subset=subset)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  
            
    
    
    
    
    
    
        
inputgenerator=generate_generator_multiple(generator=input_imgen,
                                           dir1=traindir1,
                                           dir2=traindir2,
                                           batch_size=batch_size,
                                           img_height=f,
                                           img_width=s,subset="training")       

testgenerator=generate_generator_multiple(input_imgen,
                                          dir1=testdir1,
                                          dir2=testdir2,
                                          batch_size=batch_size,
                                          img_height=f,
                                          img_width=s,subset="validation")              
          
          
  
history=final_model.fit_generator(inputgenerator,
                    #steps_per_epoch=trainsetsize/batch_size,
                    steps_per_epoch=250 ,
                    epochs = 100,
                    validation_data = testgenerator,
                    validation_steps = 100,
                    shuffle=False)
