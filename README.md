# Digital Imaging / Visual Computing

This is the repository for the Digital Imaging / Visual Computing course (05_DVC4IL) at the FH Hagenberg.

[E-Learning course](https://elearning.fh-ooe.at/course/view.php?id=34875)

Contact: [David C. Schedl](mailto:david.schedl@fh-hagenberg.at).

## Tutorials:

| #   | Tutorial (link to `.ipynb`)                             | Open in Colab                                                                                                                                                                |
| --- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Python for Computer Vision](./01_PythonTutorial.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/01_PythonTutorial.ipynb) |
| 2   | [Introduction to OpenCV](./02_Images.ipynb)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/02_Images.ipynb)         |
| 3   | [Histograms](./03_Histograms.ipynb)                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/03_Histograms.ipynb)     |
| 4   | [Filters](./04_Filters.ipynb)                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/04_Filters.ipynb)        |
| 5   | [Edges](./05_Edges.ipynb)                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/05_Edges.ipynb)          |
| 6   | [Thresh](./06_Thresh.ipynb)                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/06_Thresh.ipynb)         |
| 7   | [Lines](./07_Lines.ipynb)                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/07_Lines.ipynb)          |
| 8   | [Machine Learning](./08_ML.ipynb)                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/08_ML.ipynb)             |
| 9   | [Neural Networks](./09_NNs.ipynb)                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/09_NNs.ipynb)             |
| 10  | [Object Detection with YOLO](./10_OD.ipynb)            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/10_OD.ipynb)             |


<!--
| 11 | [Transfer Learning](./11_TL.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/11_TL.ipynb) |
| 12 | [Object Detection](./12_OD.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/12_OD.ipynb) |
-->

## Homework Tasks:

| #   | Homework (link to `.ipynb`)                           | Open in Colab                                                                                                                                                                    |
| --- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I   | [Point Operations & Histograms](./HW01_PointOps_Hists.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/HW01_PointOps_Hists.ipynb) |
| II  | [Hybrid Images](./HW02_Hybrid.ipynb)                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/HW02_Hybrid.ipynb)         |
| III | [Leaf Classification](./HW03_Leaves.ipynb)          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/HW03_Leaves.ipynb)         |
| IV  | [Hockey Dataset Analysis](./HW04_Hockey_Dataset.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/HW04_Hockey_Dataset.ipynb) |


<!--
| 4 | [Image Classification](./HW04_Classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digital-Media/di_cv/blob/main/HW04_Classification.ipynb) |
-->

## Python Setup:

Students have the option to run the code _online_ with Google Colab (requires a Google account) or _locally_ with your own installation of Python.

#### Online:

Everything runs on a Google machine, so you don't need to set up anything on your computer. Furthermore, the machines come with the most popular libraries preinstalled.
Just click on the corresponding Open in Colab badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#tutorials).

#### Local:

Install Python on your computer via [Conda/Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html) or the [Python Installer](https://www.python.org/downloads/). Use Python3, as Python2 is not supported anymore. Furthermore, you need an Editor that supports Jupyter (`.ipynb`) notebooks. I recommend using [Visual Studio Code](https://code.visualstudio.com/download). Optionally, you can also use a local server and open [Notebooks in your browser](https://test-jupyter.readthedocs.io/en/latest/install.html) (Visual Studio simplifies this).

Required packages are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

## Useful Links:

- [Python Documentation](https://docs.python.org/3.8/)
- [OpenCV Tutorial](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- If you know Matlab, you can find the differences between Matlab and Python [here](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html).

## Course Grading:

This course will be graded based on your performance in the **course homeworks**.
The [homework tasks will be announced](#Homework-Tasks) while we progress through the course.

[^1]: Using Colab is highly recommended for these tutorial(s).
