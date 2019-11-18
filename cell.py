import numpy as np
import cv2
import matplotlib.pyplot as plt


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    plt.imshow(labeled_img)
    plt.show()


img = cv2.imread('Capture1.PNG',0) # load image
margin = 3
img = img[margin:img.shape[0]-margin, margin:img.shape[1]-margin]

edges = cv2.Canny(img,30,100) # edge detection
kernel = np.ones((4,4),np.uint8)
dilation = cv2.dilate(edges, kernel,iterations = 1) # dilation to fill in cells from edge
output = np.copy(dilation)

ret, labels = cv2.connectedComponents(dilation) # label connected blobs
label, counts = np.unique(labels, return_counts=True) # number of pixels in each blob

ignore = []
threshold = 300
# remove groups with fewer pixels than threshold
for i in range(len(label)):
    if counts[i]<threshold:
        ignore.append(label[i])

for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        if labels[row, col] in ignore:
            output[row, col] = 0


imshow_components(labels) # plot grouping

fig=plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)

ax1.imshow(img,cmap = 'gray')
ax2.imshow(edges,cmap = 'gray')
ax3.imshow(dilation,cmap = 'gray')
ax4.imshow(output,cmap = 'gray')

plt.show()


