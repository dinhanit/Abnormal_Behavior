# Exam Abnormal Behavior Recognition (Data Branch).
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
        <li><a href="#Pipeline">2.1 Pipeline.</a></li>
        <li><a href="#installation">2.2 Installation.</a></li>
        <li><a href="#instructions_manual">2.3 Instructions Manual.</a></li>
      </ul>
    </li>
    </li>
    <li><a href="#development_history">Development History.</a></li>
    <li>
      <a href="#exploratory_data_analysis_eda">Exploratory Data Analysis (EDA.)</a>
      <ul>
        <li><a href="#dataframe">4.1 Dataframe.</a></li>
        <li><a href="#check_imbalanced">4.2 Check Imbalanced Data.</a></li>
        <li><a href="#kernel_den">4.3 Kernel Density Plots.</a></li>
        <li><a href="#pca">4.4 PCA.</a></li>
        <li><a href="#visual_image">4.5 Visual On Image.</a></li>
      </ul>
    <!-- </li>
    <li><a href="#development_history">Development History.</a></li> -->
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## 1. About The Branch.
<a id="about-the-branch"></a>

This is an overview of the branch directory.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/eb0aa49c-f755-44da-b07b-4be5b9c72331" width="420" height="330">


<p align="right"><a href="#top">Back to Top</a></p>


<!-- GETTING STARTED -->
## 2. Getting Started.
<a id="getting-started"></a>

This is an instructions on setting up project locally.
### 2.1 Pipeline.
<a id="Pipeline"></a>

This is our flow to run this branch.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/8e577622-29f6-46c8-b3a4-54a03c3cc030" width="420" height="330">

### 2.2 Installation.
<a id="installation"></a>

Below is all nesscesary packages that need to be installed.

1. Clone the repo.
   ```sh
   git clone https://github.com/dinhanit/Abnormal_Behavior.git
   ```
2. Install the Mediapipe packages .
   ```python
   pip install mediapipe
   ```

### 2.3 Instructions Manual.
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



<!-- CONTRIBUTING -->
## 3. Development History.
<a id="development_history"></a>


![image](https://github.com/DangLeChi/ChiTest/assets/122540817/ba2cf671-7881-4f5b-8c9f-8b70c86406a8)

<p align="right"><a href="#top">Back to Top</a></p>

<!-- USAGE EXAMPLES -->
## 4. Exploratory Data Analysis (EDA).
<a id="exploratory_data_analysis_eda"></a>

### 4.1 Dataframe.
<a id="dataframe"></a>

+ This is the original dataframe.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/e4d25dd3-a569-4571-9b90-00d09bb0dec7)

+ Anomalies of the data set.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/2b644f0b-c7d8-42ae-a412-508080d8fb5a)

+ Delete abnormal values and save dataframe.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/44024f0f-2a57-4b78-81b6-20f965e1b7f5)

### 4.2 Check imbalanced data.
<a id="check_imbalanced"></a>


<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/63439c89-8425-41ae-8d54-092b84d857bd" width="600" height="280">


### 4.3 Kernel Density Plots to visualize values in data.
<a id="kernel_den"></a> 


<!-- ![image](https://github.com/DangLeChi/ChiTest/assets/122540817/a26254e3-b810-457d-9791-a1800014450d) -->
<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/a26254e3-b810-457d-9791-a1800014450d" width="500" height="290">


<!-- ![image](https://github.com/DangLeChi/ChiTest/assets/122540817/e37f89c6-2433-4a82-96bd-48e6519a0afc) -->
<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/e37f89c6-2433-4a82-96bd-48e6519a0afc" width="500" height="290">


### 4.4 PCA.
<a id="pca"></a>


<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/7cf42411-0d25-4551-8b9e-1908d167e0b0" width="370" height="370">



+ Histogram for Label in pca_dataframe.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/9e86edf7-5dd3-4098-b14b-92589d06c7eb" width="580" height="230">

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/4541c132-3139-4ce3-9cde-b83dccc7e704" width="580" height="230">

+ Visualize PCA.

![gif_train](visual_pca_train.gif)
![gif_test](visual_pca_test.gif)

### 4.5 Visualize on Image.
<a id="visual_image"></a>

+ Visualize selected keypoints on image.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/1c317e33-0533-4d29-8a4d-652bda25d4e6" width="400" height="300">

+ Visual distances between selected keypoints on image.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/431bb2a9-16c6-464f-b603-5a3bcfcd2c46" width="400" height="300">

+ Visualize selected keypoints in 3D.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/01540066-b940-4611-819d-3dc39df0b005" width="400" height="300">

+ Visualize all keypoints on 3D.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/ec61d54a-4999-43b2-a98e-02a5937afa2b" width="350" height="300">


<p align="right"><a href="#top">Back to Top</a></p>










