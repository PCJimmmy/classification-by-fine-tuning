The project is to build a classification model using pre-trained model.
In this project I have used InceptionV3. 
<br>
One can use any pre-trained model.
<br>
Using pre-trained model to create model for your own task is called transfer learning as you are tranferring the 
learnings of previous model for your own dataset.

How to make it work : 
* run command **python bin/training.py** to start training on your dataset. There are two options available:
    * fine_tune = False. In this case only the last convolution layer is trained rest inception model is set to 
    trainable = False
    * fine_tune = True. In this case last added convolution layer along with the last InceptionV3 block is trained.
    <br>
*fine_tune parameter can be set in training.py file.*
* before training you need 3 files. _classes.txt_, _training_images_path.txt_ and _validation_images_path.txt_ inside 
util_files 
folder. Script to create these files are available.
    * classes.txt can be created using **python extra_help/read_classes.py**
    * training and validation images path files can be created by running the script **python utils/create_data.py**
  
_P.S. :_<br>
* set config.yaml file values accordingly
*  data should be prepared in given fashion. parent_dir -> label -> images_in_jpg_format. Say you have two classes 
dog and cat. Then all the images of dog should be in dog folder and similar for cat.
