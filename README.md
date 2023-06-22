# READ ME FIRST 

# ICUB ROBOT CODE 

# This map contains the coding part of my thesis "Deep Learning-based Object Recognition Model for Humanoid Robot". 

# The goal of my thesis was to examine if the use of visual inputs from multiple sensors enhances recognition accuracy compared to deep
# neural networks that use single visual sensory inputs. In addition to the deep neural network models, an explainable artificial intelligence
# algorithm was utilised to explain the reasoning of the models. The proposed multiple sensor object recognition model can be used for 
# real world applications by robots with similar sensors (as the robot used in this thesis). 

# The images of the thesis were made by the iCub robot which was equipped with three RGB cameras (two as the eyes on the robot and one camera
# on the forehead). The images contain 210 daily life objects. This means that the models were trained on 48360 images. The objects 
# were placed on a turntable and every five percent rotation an image was taken. 

# The coding files: 
# For every camera there was a preprocessing stage, therefore there are three files with the name Preprocessing. 
# For every camera there was a training fase. Therefore there are three training files. 
# The training for each camera was improved through hyperparameter tuning. Consequently, there are three hyperparameter tuning files. 
# Every model is validated. These files are named Validation Left, Realsense and Right. 
#  
