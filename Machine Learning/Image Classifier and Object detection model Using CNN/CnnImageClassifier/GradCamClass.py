from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class GradCamClass:
    
    # load and prepare the image
    @staticmethod
    def load_image(filename):
        # load the image
        img = load_img(filename, target_size=(256, 256))
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 3 channels
        #img = img.reshape(1, 32, 32, 3)
        img=np.expand_dims(img, axis=0)
        img = img.reshape(1,256,256,3)
        #img=preprocess_input(img)
        # prepare pixel data
        #img = img.astype('float32')
        #img = img / 255.0
        return img
    
    # predict image label and draw boudning box
    @staticmethod
    def predictAndLocalizeObject(model,imagePath):
        #load image for prediction
        imageObj = GradCamClass.load_image(imagePath)
        #predict object class in the image with the CNN trained model
        predictedOutcome = model.predict_classes(imageObj)
        print("predicted class label is : ",predictedOutcome[0])
        np.argmax(predictedOutcome[0])
        #obtain prediction map of the predicted class
        predictedClassDefaultMap= model.output[:, predictedOutcome[0]]
        #extract last convolution layer of the neural network
        last_conv_layer = model.get_layer('conv2d_6')
        #obtain gradient predicted class map and output of the last concolution layer
        gradientObj = K.gradients(predictedClassDefaultMap, last_conv_layer.output)[0] 
        #obtain pool gradients
        pooled_grads = K.mean(gradientObj, axis=(0, 1, 2))
        print(last_conv_layer.output[0])
        iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([imageObj])
        print("length of arrays : ",pooled_grads_value.shape,conv_layer_output_value.shape)
        
        #iterate over all the values of image vector
        for i in range(256):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        #generate heatmap from the output of of the last convolution layer
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        #plot the heat map
        plt.matshow(heatmap)
        
        #read the original image with CV
        originalImageObj = cv2.imread(imagePath)
        #resize the heatmap according to the original image size
        heatmap = cv2.resize(heatmap, (originalImageObj.shape[1], originalImageObj.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #get the current working directory
        current_working_directory = os.getcwd()
        temp_Image_Heatmap_Path = current_working_directory +"/image_Heatmap_Temp.jpg"
        #save the heat map in temporary location
        cv2.imwrite(temp_Image_Heatmap_Path, heatmap* 0.4)
        # Grayscale then Otsu's threshold
        # read the saved heatmap
        heatMapImageObj = cv2.imread(temp_Image_Heatmap_Path)
        #save the original image in a temp location for processing
        temp_Image_Path=current_working_directory +"/originalImageTemp.jpg"
        cv2.imwrite(temp_Image_Path,originalImageObj)
        #obgain object of originalimage saved in temporary location
        tempImageObj = cv2.imread(temp_Image_Path)
        # Grayscale then Otsu's threshold
        gray = cv2.cvtColor(heatMapImageObj, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imshow('thresh', thresh)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            #draw bounding box
            cv2.rectangle(tempImageObj, (x, y), (x + w, y + h), (36,255,12), 2)
        
        #save the bound boxed original Image copy
        cv2.imwrite(temp_Image_Path,tempImageObj)
        cv2.destroyAllWindows()

        #superimposed_img = image+ img
        #cv2.imwrite('/Users/vikas/Documents/model/elephant_cam.jpg', superimposed_img)
    
    