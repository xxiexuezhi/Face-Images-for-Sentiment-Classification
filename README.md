# Face-Images-for-Sentiment-Classification
 To build a CNN and apply transfer-learning to a new classification problems.  this is a private kaggle dataset, the dataset link is here: https://drive.google.com/drive/folders/1HtQkw7FiK9BT881teXnGj5_piibBMHdW?usp=sharin. This dataset consists of about 28000 images. Each example is 48x48 grayscale image, associated with a label from 7  sentiments or classes (like angry,  happy, and so on)

The  preprocessing steps that reshape the data into 48*48  and resize(crop) the image to 32 * 32 were done. I used the rotation for the randomness for data augmentation. 

I designed CNN model with three hidden layers. The kernel size of all 3 layers are the same (3*3). The total parameters my model learned is 107015


Hyperparameters setting:  epochs=30, batch-size=256, lr (learning rate for Adam)=.0001
I trained the model with several different epochs and batch-size, and I selected the one giving highest accuracy rate

The model I used for pre-trained CNN model is vgg-16. VGG-16 is a convolutional neural network and the network is 16 layers deep and 5 blocks. VGG-16 is trained on more than a million images from the ImageNet database. As a result, the network has learned rich feature representations for a wide range of images. I used the vgg-16 provided by Keras. VGG-16 is a good example for transfer learning. I used  global-average-pooling layer, then use dense layer to predict the result. Also, I freezed all the bottleneck layers.    

My model is trained on Training set. I set epochs to {20, 30, 40}, batch_size {128, 256} I  test the model performance on separated test datasets.
   	
For the Vgg-16 pretrained model, I freeze all block layers and only train the dense layer with the training data. I set epochs to {20, 30, 40}, batch_size {128, 256}. Then test the performance on test datasets

