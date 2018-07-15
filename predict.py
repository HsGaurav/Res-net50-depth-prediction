import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import scipy.misc
import models

def predict(model_data_path, image_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = scipy.misc.toimage(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        imo = pred[0,:,:,0]
        
    return imo   

def main():
    
    model_path = 'NYU_FCRN.ckpt'
    video_capture = cv2.VideoCapture(0)
    ret = True
    x = 159             #x and y are the value of the pixel where you want the value of rgb
    y = 127
    starttime = time.time()
    
    while ret:
        ret, frame = video_capture.read()
        hola = predict(model_path, frame)
 
        print('-------------',hola.shape,'--------------')
        fig = plt.figure()
        ii = plt.imshow(hola, interpolation='nearest')
        print(ii)
        fig.colorbar(ii)
        
        #value of the rgb image at the give pixel of x and y
        im = scipy.misc.toimage(hola)
        pix = im.load()
#         print(im.size)
        print(pix[x,y])
        
        plt.show()
        tf.reset_default_graph()
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        print("--- %s seconds ---" % (time.time() - starttime))
        
    video_capture.release()
    cv2.destroyAllWindows()
    os._exit(0)

if __name__ == '__main__':
    main()
