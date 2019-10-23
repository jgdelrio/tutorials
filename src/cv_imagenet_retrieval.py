from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib


page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289")#ship synset
print(page.content)
# BeautifulSoup is an HTML parsing library
soup = BeautifulSoup(page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line

bikes_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02834778")#bicycle synset
print(bikes_page.content)
# BeautifulSoup is an HTML parsing library
from bs4 import BeautifulSoup
bikes_soup = BeautifulSoup(bikes_page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line

# Split urls in diff lines
str_soup=str(soup)#convert soup to string so it can be split
type(str_soup)
split_urls=str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(split_urls))#print the length of the list so you know how many urls you have

bikes_str_soup=str(bikes_soup)#convert soup to string so it can be split
type(bikes_str_soup)
bikes_split_urls=bikes_str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(bikes_split_urls))

# Create folders:
# !mkdir /content/train #create the Train folder
# !mkdir /content/train/ships #create the ships folder
# !mkdir /content/train/bikes #create the bikes folder
# !mkdir /content/validation
# !mkdir /content/validation/ships #create the ships folder
# !mkdir /content/validation/bikes #create the bikes folder



# Correct shape, format and store them
img_rows, img_cols = 32, 32  # number of rows and columns to convert the images to
input_shape = (img_rows, img_cols, 3)  # format to store the images (rows, columns,channels) called channels last


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


n_of_training_images = 100  # the number of training images to use
for progress in range(n_of_training_images):  # store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if (progress % 20 == 0):
        print(progress)
    if not split_urls[progress] == None:
        try:
            I = url_to_image(split_urls[progress])
            if (len(I.shape)) == 3:  # check if the image has width, length and channels
                save_path = '/content/train/ships/img' + str(progress) + '.jpg'  # create a name of each image
                cv2.imwrite(save_path, I)
        except:
            None

# do the same for bikes:
for progress in range(n_of_training_images):  # store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if (progress % 20 == 0):
        print(progress)
    if not bikes_split_urls[progress] == None:
        try:
            I = url_to_image(bikes_split_urls[progress])
            if (len(I.shape)) == 3:  # check if the image has width, length and channels
                save_path = '/content/train/bikes/img' + str(progress) + '.jpg'  # create a name of each image
                cv2.imwrite(save_path, I)
        except:
            None

# Validation data:
for progress in range(50):  # store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if (progress % 20 == 0):
        print(progress)
    if not split_urls[progress] == None:
        try:
            I = url_to_image(split_urls[
                                 n_of_training_images + progress])  # get images that are different from the ones used for training
            if (len(I.shape)) == 3:  # check if the image has width, length and channels
                save_path = '/content/validation/ships/img' + str(progress) + '.jpg'  # create a name of each image
                cv2.imwrite(save_path, I)
        except:
            None
# do the same for bikes:
for progress in range(50):  # store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if (progress % 20 == 0):
        print(progress)
    if not bikes_split_urls[progress] == None:
        try:
            I = url_to_image(bikes_split_urls[
                                 n_of_training_images + progress])  # get images that are different from the ones used for training
            if (len(I.shape)) == 3:  # check if the image has width, length and channels
                save_path = '/content/validation/bikes/img' + str(progress) + '.jpg'  # create a name of each image
                cv2.imwrite(save_path, I)
        except:
            None

# print("\nTRAIN:\n")
# print("\nlist the files inside ships directory:\n")
# !ls / content / train / ships  # list the files inside ships
# print("\nlist the files inside bikes directory:\n")
# !ls / content / train / bikes  # list the files inside bikes
# print("\nVALIDATION:\n")
# print("\nlist the files inside ships directory:\n")
# !ls / content / validation / ships  # list the files inside ships
# print("\nlist the files inside bikes directory:\n")
# !ls / content / validation / bikes  # list the files inside bikes