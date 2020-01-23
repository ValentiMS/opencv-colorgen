import cv2
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial import KDTree
import webcolors
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("-i", "--image", help= "path to the image file", required=True)
ap.add_argument("-c", "--colors", help= "number of colors to be detected", default=5, type=int)
args = vars(ap.parse_args())

IMAGE_SOURCE = args["image"]
INTERNAL_SHAPE = (600, 400)
COLOR_NUMBER = args["colors"]

def css3_to_rgb():
	names = []
	rgb = []
	
	for hex, name in webcolors.css3_hex_to_names.items():
		names.append(name)
		rgb.append(webcolors.hex_to_rgb(hex))
	
	return names, rgb

css3_rgb = css3_to_rgb()	
def nearest_color(rgb_tuple):
	colordb = KDTree(css3_rgb[1])
	_, index = colordb.query(rgb_tuple)
	return css3_rgb[0][index]
		
image = cv2.imread(IMAGE_SOURCE)
print("Loaded image {} of dimensions {}".format(IMAGE_SOURCE, image.shape))

#Convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

resized_image = cv2.resize(image, INTERNAL_SHAPE, interpolation=cv2.INTER_AREA)
resized_image = resized_image.reshape(resized_image.shape[0] * resized_image.shape[1], 3)

clf = KMeans(COLOR_NUMBER)
labels = clf.fit_predict(resized_image)
counts = Counter(labels)
center_colors = clf.cluster_centers_
rgb_colors = [center_colors[i] for i in counts.keys()]

detected_colors = []
for i in rgb_colors:
	rgb_color = tuple(i)
	approximated_color = nearest_color(rgb_color)
	detected_colors.append(approximated_color)

output_colors = ", ".join(detected_colors)
print("Detected colors are: {}".format(output_colors))