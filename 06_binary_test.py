# %% [markdown]
# # Tutorial 06 - Binary
#
# ## Dr. David C. Schedl
#
# Note: this tutorial is geared towards students **experienced in programming** and aims to introduce you to **Digital Imaging / Computer Vision** techniques.
#

# %% [markdown]
# # Table of Contents
#
# - Thresholding
#

# %% [markdown]
# # Initilization

# %% [markdown]
# As always let's import useful libraries, first.

# %%
import os
import cv2  # openCV
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# %% [markdown]
# We will work with images today. So let's download some with `curl` (the same sources as in `02_OpenCV.ipynb`).

# %%
# !curl -o "cat.jpg" "https://placekitten.com/320/320" --silent
# !curl -o "gogh.jpg" "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Vincent_van_Gogh_-_National_Gallery_of_Art.JPG/367px-Vincent_van_Gogh_-_National_Gallery_of_Art.JPG" --silent
# !curl -o "einstein.jpg" "https://www.cns.nyu.edu/~lcv/ssim/index_files/image003.jpg" --silent
# !curl -o "woman.jpg" "https://live.staticflickr.com/8859/18045025168_3a1ffa6521_c_d.jpg" --silent
# !curl -o "road110.png" "https://storage.googleapis.com/kagglesdsdata/datasets/671172/1181356/images/road110.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221024%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221024T141558Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4caf2429c7705e2a061941aece256a66296c5b4f7e28e427dcffd05d7eb720665a5619bfb139010c194fa31152f18d0dcc02b6aec87f4e19c614f726b9869acd9e6c2e3c716336ab6fd17dabdccd85d5fd832b2e1b5b46c241994d033ba340d4c5e7e4179903b78efa67ee9a8837606f6971612fc69acb2380f2c28aabeeb0ae0721c89c5dbf3cc0348bb5c3752c2ad8c341d61f8e3de78e8bf61a68e325024caf13b3ed2fc3957aa882fbe3029d2bb8c8d45bbb607043ec1f3594ad18a1de3795cc3577abc78c27957a15edeedba0c3eb9232d9252e686bdf04376aec8e34da7f074ee3d39bab6e8064bd0f007dfed69661bff1bc49a8019e0a9b2e0ff14344"
# !curl -o "sudoku.png" "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/sudoku.png" --silent

# %% [markdown]
# # Thresholding

# %% [markdown]
# Let's first look at our sample image showing multiple coins on a fairly uniform background.

# %%
from skimage import data

coins = data.coins()

fig = plt.figure(figsize=(12, 7), facecolor="white")
plt.subplot(1, 2, 1), plt.imshow(coins, cmap="gray")
plt.title(f"Image {coins.shape[::-1]}"), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.hist(coins.ravel(), bins=256, range=[0, 255])
plt.title("Histogram"), plt.yticks([])
plt.show()

# %% [markdown]
# ## Global Thresholding
#
# Let's first look at a global thresholding approach.
# All pixels with a value above the threshold are True, all pixels with a value below the threshold are False.
# Can you come up with a threshold value that separates the coins from the background? <br>
# You can also try to blur the image. Does this help?

# %%
# a slider for changing the threshold
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, IntSlider, FloatSlider
from skimage import data

img = data.coins()
print(img.shape)


def plot_threshold(threshold, blur=0):
    _img = img.copy()

    if blur > 0:
        _img = cv2.GaussianBlur(_img, None, blur)

    T = _img > threshold

    # plot the thresholded image and the histogram
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1), plt.imshow(T, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Threshold ({T.sum()/T.size*100:3.2f}% pixels selected)"), plt.xticks(
        []
    ), plt.yticks([])
    plt.subplot(1, 2, 2),
    plt.hist(_img[T].ravel(), range=[0, 255], bins=256),
    plt.hist(_img[np.invert(T)].ravel(), range=[0, 255], bins=256)
    plt.show()
    # return fig


interact(
    plot_threshold,
    threshold=IntSlider(min=0, max=255, step=1, value=128),
    blur=FloatSlider(min=0, max=5, step=0.1, value=0),
)

# %% [markdown]
# ## Local Thresholding
#
# Let's now look at a local thresholding approach.
# In comparison to the global thresholding, the local thresholding is applied to small regions of the image.
# The region size is defined by the $w$ parameter.

# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import data


def adaptive_threshold(w=11, c=2, blur=0):
    img = data.coins()

    if blur > 0:
        img = cv2.GaussianBlur(img, None, blur)

    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, w, c
    )
    th3 = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, w, c
    )

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    titles = [
        "Original Image",
        "Global Thresholding ($q$ = 127)",
        "Adaptive Mean Thresholding",
        "Adaptive Gaussian Thresholding",
    ]
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


interact(
    adaptive_threshold,
    w=IntSlider(min=3, max=255, step=2, value=51),
    c=IntSlider(min=-50, max=50, step=1, value=2),
    blur=FloatSlider(min=0, max=5, step=0.1, value=0),
)
