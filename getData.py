import os
import scipy.io as sio
import  matplotlib.image as readImage
import  matplotlib.pyplot as plt
import numpy as np
def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".png")]
        for f in file_names:
            images.append(readImage.imread(f))
            labels.append(int(d))
    return images, labels
ROOT_PATH = "/home/lijq/IdeaProjects/classAreticle/resize"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
t_images, t_labels = load_data(test_data_directory)
sio.savemat('./feature.mat',{"train_feature":images,"train_labels":labels
                                     , "test_feature":t_images,"test_label":t_labels})
# print np.array(labels).size
# plt.hist(labels,62)
# plt.show()

# traffic_signs = [300, 2250, 3650, 4000]
#
# # Fill out the subplots with the random images that you defined
# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)
#     print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
#                                                   images[traffic_signs[i]].min(),
#                                                   images[traffic_signs[i]].max()))
# plt.show()

#Get the unique labels
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image
    plt.imshow(image)

# Show the plot
plt.show()





