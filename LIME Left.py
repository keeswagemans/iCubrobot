# Loading libraries

import pickle
from lime import lime_image
import tensorflow as tf
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Loading files

X_test_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_left.pkl', 'rb'))
Y_test_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_left.pkl', 'rb'))
print(len(X_test_left))

# Loading model

model_left = tf.keras.models.load_model('CNN_1_left')

# Explainer

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(X_test_left[2627].astype('double'),
                                         model_left.predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000)

temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=True)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=False,
                                                num_features=10,
                                                hide_rest=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
ax1.imshow(mark_boundaries(temp_1, mask_1).astype('uint8'))
ax2.imshow(mark_boundaries(temp_2, mask_2).astype('uint8'))
ax1.axis('off')
ax2.axis('off')
plt.show()
