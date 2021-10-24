# USAGE
# python quant.py  --clusters 8 --image images/worldcup.jpg --saveimage images/worldcup-quant.jpg
# python quant.py  --clusters 8 --image images/nature.png --saveimage images/nature-quant.png
# python quant.py  --clusters 4 --image images/backgroundcut_highres.png --saveimage images/erichires-quant.png
# python quant.py  --clusters 4 --image images/backgroundcut_600.png --saveimage images/eric600-quant.png
# python quant.py  --clusters 4 --image images/backgroundcut_600.jpg --saveimage images/eric600-quant.jpg

# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2

# import numpy as np
import streamlit as st
# import cv2
from  PIL import Image, ImageEnhance
from urllib.request import urlopen

Output_image = 500
quantization_colors = 4;

def main():
    @st.cache
    def load_image(url):
        with urlopen(url) as response:
            image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]] # BGR -> RGB
        return image

    image = load_image('https://raw.githubusercontent.com/rafaelgrecco/Filter-Selector/master/Images/placeholder.jpg')
    image = Image.fromarray(image)
    st.title('Filter Selector')
    st.sidebar.title('Sidebar')

    menu = ['Filters','Image Corrections', 'Information']
    op = st.sidebar.selectbox('Option', menu)

    if op == 'Filters':

        img = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

        if img is not None:
            image = Image.open(img)
            st.sidebar.text('Original Image')
            st.sidebar.image(image, width=200)

        filters = st.sidebar.radio('Filters', ['Original','Quantization','Grayscale', 'Sepia', 'Blur', 'Contour', 'Sketch'])

        if filters == 'Quantization':
            img_convert = np.array(image.convert('RGB'))

            # load the image and grab its width and height
            # image = cv2.imread(img)
            image = img_convert
            (h, w) = image.shape[:2]

            # convert the image from the RGB color space to the L*a*b*
            # color space -- since we will be clustering using k-means
            # which is based on the euclidean distance, we'll use the
            # L*a*b* color space where the euclidean distance implies
            # perceptual meaning
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # reshape the image into a feature vector so that k-means
            # can be applied
            image = image.reshape((image.shape[0] * image.shape[1], 3))

            # apply k-means using the specified number of clusters and
            # then create the quantized image based on the predictions
            clt = MiniBatchKMeans(n_clusters = quantization_colors)
            labels = clt.fit_predict(image)
            quant = clt.cluster_centers_.astype("uint8")[labels]

            # reshape the feature vectors to images
            quant = quant.reshape((h, w, 3))
            image = image.reshape((h, w, 3))

            # convert from L*a*b* to RGB
            quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

            # display the images and wait for a keypress
            # cv2.imwrite(args["saveimage"], quant)
            # cv2.imshow("image", np.hstack([image, quant]))
            st.image(quant, width=Output_image)

        elif filters == 'Grayscale':
            img_convert = np.array(image.convert('RGB'))
            gray_image = cv2.cvtColor(img_convert, cv2.COLOR_RGB2GRAY)
            st.image(gray_image, width=Output_image)
            
        elif filters == 'Sepia':
            img_convert = np.array(image.convert('RGB'))
            img_convert = cv2.cvtColor(img_convert, cv2.COLOR_RGB2BGR)
            kernel = np.array([[0.272, 0.534, 0.131],
                            [0.349, 0.686, 0.168],
                            [0.393, 0.769, 0.189]])
            sepia_image = cv2.filter2D(img_convert, -1, kernel)
            st.image(sepia_image, channels='BGR', width=Output_image)
        
        elif filters == 'Blur':
            img_convert = np.array(image.convert('RGB'))
            slide = st.sidebar.slider('Quantidade de Blur', 3, 81, 9, step=2)
            img_convert = cv2.cvtColor(img_convert, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(img_convert, (slide,slide), 0, 0)
            st.image(blur_image, channels='BGR', width=Output_image) 
        
        elif filters == 'Contour':
            img_convert = np.array(image.convert('RGB'))
            img_convert = cv2.cvtColor(img_convert, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(img_convert, (11,11), 0)
            canny_image = cv2.Canny(blur_image, 100, 150)
            st.image(canny_image, width=Output_image)

        elif filters == 'Sketch':
            img_convert = np.array(image.convert('RGB')) 
            gray_image = cv2.cvtColor(img_convert, cv2.COLOR_RGB2GRAY)
            inv_gray = 255 - gray_image
            blur_image = cv2.GaussianBlur(inv_gray, (25,25), 0, 0)
            sketch_image = cv2.divide(gray_image, 255 - blur_image, scale=256)
            st.image(sketch_image, width=Output_image) 
        else: 
            st.image(image, width=Output_image)

    if op == 'Image Corrections':

        img = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])
        
        if img is not None:
            image = Image.open(img)
            st.sidebar.text('Original Image')
            st.sidebar.image(image, width=200)

        MImage = st.sidebar.radio('Image enhancement', ['Original', 'Contrast', 'brightness', 'Sharpness'])

        if MImage == 'Contrast':
            slide = st.sidebar.slider('Contrast', 0.0, 2.0, 1.0)
            enh = ImageEnhance.Contrast(image)
            contrast_image = enh.enhance(slide)
            st.image(contrast_image, width=Output_image)
        
        elif MImage == 'brightness':
            slide = st.sidebar.slider('brightness', 0.0, 5.0, 1.0)
            enh = ImageEnhance.Brightness(image)
            brightness_image = enh.enhance(slide)
            st.image(brightness_image, width=Output_image)

        elif MImage == 'Sharpness':
            slide = st.sidebar.slider('Sharpness', 0.0, 2.0, 1.0)
            enh = ImageEnhance.Sharpness(image)
            sharpness_image = enh.enhance(slide)
            st.image(sharpness_image, width=Output_image)
        else: 
            st.image(image, width=Output_image)
    
    elif op == 'Information':
        st.subheader('Project developed by Rafael Messias Grecco')



if __name__ == '__main__':
    main()









# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# ap.add_argument("-c", "--clusters", required = True, type = int,
#   help = "# of clusters")
# ap.add_argument("-s", "--saveimage", required=True,
#     help="path to save image")
# args = vars(ap.parse_args())

# # load the image and grab its width and height
# image = cv2.imread(args["image"])
# (h, w) = image.shape[:2]

# # convert the image from the RGB color space to the L*a*b*
# # color space -- since we will be clustering using k-means
# # which is based on the euclidean distance, we'll use the
# # L*a*b* color space where the euclidean distance implies
# # perceptual meaning
# image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # reshape the image into a feature vector so that k-means
# # can be applied
# image = image.reshape((image.shape[0] * image.shape[1], 3))

# # apply k-means using the specified number of clusters and
# # then create the quantized image based on the predictions
# clt = MiniBatchKMeans(n_clusters = args["clusters"])
# labels = clt.fit_predict(image)
# quant = clt.cluster_centers_.astype("uint8")[labels]

# # reshape the feature vectors to images
# quant = quant.reshape((h, w, 3))
# image = image.reshape((h, w, 3))

# # convert from L*a*b* to RGB
# quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
# image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# # display the images and wait for a keypress
# cv2.imwrite(args["saveimage"], quant)
# cv2.imshow("image", np.hstack([image, quant]))
# cv2.waitKey(0)
