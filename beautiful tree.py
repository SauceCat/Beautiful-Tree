import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# the number of major colors you want to keep
n_color = 10

# load the image
image_path = "/Users/mac/Desktop/termtwo/self-page/beautiful tree/goo02.jpg"
im = Image.open(image_path)
pix = np.array(im.getdata(), np.float64) / 255

# do kmeans cluster based on the number of major colours you want to keep
kmeans = KMeans(n_clusters=n_color, init="random", random_state=1).fit(pix)
labels = kmeans.predict(pix)
main_colors = kmeans.cluster_centers_
w, h = im.size

# recreate the image based on the major colours
def recreate_image(main_colors, labels, w, h):
    image = np.zeros((h, w, 3))
    label_idx = 0
    for i in range(h):
        for j in range(w):
            image[i][j] = main_colors[labels[label_idx]]
            label_idx += 1
    return image

# show the image
plt.imshow(recreate_image(main_colors,labels, w, h))

# get the x, y axis values of all pixels
positions = []
for i in range(h):
    for j in range(w):
        positions.append([i, j])

# run the tree model to generate the color blocks
from sklearn import tree
X = np.array(positions, dtype=np.float64)
Y = np.array(labels, dtype=np.float64)
clf = tree.DecisionTreeClassifier(random_state=0, max_depth=15)
clf = clf.fit(X, Y)
new = clf.predict(X)

# show the colour blocks
plt.figure(facecolor='white')
fig = plt.imshow(recreate_image(main_colors,new, w, h))
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.axis('off')
plt.savefig("/Users/mac/Desktop/termtwo/self-page/beautiful tree/goo0215.png",bbox_inches='tight',pad_inches = 0)

