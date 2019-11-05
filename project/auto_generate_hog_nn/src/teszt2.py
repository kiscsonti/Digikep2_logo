from skimage.feature import hog
from skimage import data
import cv2
import numpy as np

image = cv2.imread("/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/test/apple/image_512x512_2019-10-29_13-43-47-6676740-513.png", cv2.IMREAD_GRAYSCALE)

newimg = cv2.resize(image, (int(256), int(256)))

fd = hog(newimg, orientations=9, pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2), visualize=False, multichannel=False, feature_vector=False)

print(type(fd))
print(newimg.shape)
print(fd.shape)


print(np.reshape(fd, (63*2, 63*2, 9)).shape)



# import matplotlib.pyplot as plt
#
# from skimage.feature import hog
# from skimage import data, exposure
#
#
# image = data.astronaut()
#
# fd, hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
#                     cells_per_block=(2, 2), visualize=True, multichannel=True, )
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()