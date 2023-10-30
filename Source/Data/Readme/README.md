# Exam Abnormal Behavior Recognition (Data Branch)
<p id="top"></p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents.</summary>
  <ol>
    <li>
      <a href="#about-the-branch">About The Branch.</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started.</a>
      <ul>
        <li><a href="#Pipeline">Pipeline.</a></li>
        <li><a href="#installation">Installation.</a></li>
        <li><a href="#instructions_manual">Instructions Manual.</a></li>
      </ul>
    </li>
    <li>
      <a href="#exploratory_data_analysis_eda">Exploratory Data Analysis (EDA.)</a>
      <ul>
        <li><a href="#dataframe">3.1 Dataframe.</a></li>
        <li><a href="#check_imbalanced">3.2 Check Imbalanced Data.</a></li>
        <li><a href="#kernel_den">3.3 Kernel Density Plots.</a></li>
        <li><a href="#pca">3.4 PCA.</a></li>
        <li><a href="#visual_image">3.5 Visual On Image.</a></li>
      </ul>
    </li>
    <li><a href="#development_history">Development History.</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Branch
<a id="about-the-branch"></a>

This is an overview of the branch directory

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/fa020426-8549-4df6-bd50-ca8b81d10de0)


<p align="right"><a href="#top">Back to Top</a></p>


<!-- GETTING STARTED -->
## Getting Started.
<a id="getting-started"></a>

This is an instructions on setting up project locally.
### Pipeline
<a id="Pipeline"></a>

This is our flow to run this branch
![image](https://github.com/DangLeChi/ChiTest/assets/122540817/f7248465-d2c7-494b-921f-c4e73e3096b7)

### Installation.
<a id="installation"></a>

Below is all nesscesary packages that need to be installed

1. Clone the repo
   ```sh
   git clone https://github.com/dinhanit/Abnormal_Behavior.git
   ```
2. Install the Mediapipe packages 
   ```python
   pip install mediapipe
   ```

### Instructions Manual.
<a id="instructions_manual"></a>

1. As above pipeline charts, you need to collect data first. Open folder `DataSets`, run `Collect_Image_2.py`
   ```python
   python Collect_Image_2.py
   ```
   After run that file, we will have images that will be data in `Origin` folder (we have already collected 800 images, namely 400/class). Then, run `SplitData.py`
   ```
   python SplitData.py
   ```
   It will split all images from `Origin` folder to `SplitData` folder (with ratio 8/2, we split 638 images for train set and 162 images for test set).

2. Back to `Data` folder and run `Pre_Data`
   ```sh
   python Pre_Data.py
   ```
   This code will call `Pre_Image.py` (Pre_Image return the landmarks from an image). `Pre_Data.py` will take all images from `SplitData/train` folder and get landmarks from all images, and save them to `Prepared_data_train.csv` files. 
   Then, this files will have 638 rows represent for 638 images in train set. Similarly, `Prepared_data_test.csv` will contain 162 rows for 162 images in test set.

3. Run `EDA_data.ipynb`, return the numpy array data for tranning. 

<p align="right"><a href="#top">Back to Top</a></p>



<!-- USAGE EXAMPLES -->
## Exploratory Data Analysis (EDA).
<a id="exploratory_data_analysis_eda"></a>

### Dataframe.
<a id="dataframe"></a>

+ This is the original dataframe.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/e7925f09-7a99-4859-88c1-290ce154bb0b)

+ Anomalies of the data set.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/aea78a26-eac8-4fc9-a4e5-aa07dd6d03a1)

+ Delete abnormal values and save dataframe

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/cd739557-7f0d-4d0c-a3d8-4a99df0a768d)

### Check imbalanced data.
<a id="check_imbalanced"></a>


![image](https://github.com/DangLeChi/ChiTest/assets/122540817/87a8c178-ee09-4a70-b100-8cf87faae33e)

### Kernel Density Plots to visualize values in data.
<a id="kernel_den"></a> 


![image](https://github.com/DangLeChi/ChiTest/assets/122540817/734d5a8a-832b-46d3-aa58-2906c14784bb)
![image](https://github.com/DangLeChi/ChiTest/assets/122540817/97b0c8e6-2249-4cf9-9768-90478df4d6bd)

### PCA.
<a id="pca"></a>


![image](https://github.com/DangLeChi/ChiTest/assets/122540817/25cfac25-d77e-4c34-a3e4-8a62ee5565f9)


+ Histogram for Label in pca_dataframe.
![image](https://github.com/DangLeChi/ChiTest/assets/122540817/6a72c0a6-ade5-44e5-8057-552243452981)
![image](https://github.com/DangLeChi/ChiTest/assets/122540817/5123e25c-9502-4e0e-8fc2-70133d96da81)

+ Visualize PCA.
![gif_train](video_train.gif)
![gif_test](video_test.gif)

### Visualize on Image.
<a id="visual_image"></a>

+ Visual keypoints on image.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/64a90fb1-78c9-4012-862a-1f2e193ab120)

+ Visual distance keypoints on image.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/e0f67a2c-439e-4e07-ad49-8083d06b4226)

+ Visual line image on 2D.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/d8f94b6b-a8b3-41c0-b3ee-33b0e0bd8c1a)

+ Visual keypoints on 3D.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/9d66fef2-e818-43ec-a947-8b92b41f94ff)

+ Visual all keypoint on 3D.
![image](https://github.com/DangLeChi/ChiTest/assets/122540817/188158aa-b26b-4a1a-8a23-5ad6e512e158)


<p align="right"><a href="#top">Back to Top</a></p>



<!-- CONTRIBUTING -->
## Development History.
<a id="development_history"></a>

![history](Readme\history.png)
<p align="right"><a href="#top">Back to Top</a></p>






