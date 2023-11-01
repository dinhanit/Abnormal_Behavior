# Exam Abnormal Behavior Recognition (Data Branch).

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
    <li><a href="#choose_keypoints">How to choose keypoints?</a></li>
    <li>
      <a href="#exploratory_data_analysis_eda">Exploratory Data Analysis (EDA.)</a>
      <ul>
        <li><a href="#dataframe">5.1 Dataframe.</a></li>
        <li><a href="#check_imbalanced">5.2 Check Imbalanced Data.</a></li>
        <li><a href="#kernel_den">5.3 Kernel Density Plots.</a></li>
        <li><a href="#pca">5.4 PCA.</a></li>
        <li><a href="#visual_image">5.5 Visual On Image.</a></li>
      </ul>

  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## 1. About The Branch.
<a id="about-the-branch"></a>

### Examination regulations:

- Student's laptop mush have camera, and it can work
- While taking exam, don't cover the camera. If you cover, it means you're cheating 

### Abnormal Criteria:
- As above, cover the camera
- Look outside (up/down/left/right) too long (over 10s)

This is an overview of the branch directory.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/eb0aa49c-f755-44da-b07b-4be5b9c72331" width="440" height="350">

<!-- GETTING STARTED -->
## 2. Getting Started.
<a id="getting-started"></a>

This is an instructions on setting up project locally.

### 2.1 Pipeline.
<a id="Pipeline"></a>

This is our flow to run this branch.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/90c04253-513d-4b5d-a3d9-557240e07a6d" width="440" height="350">

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

1. As above pipeline charts, you need to collect data first. Open folder `DataSets`, run `CollectData2.py`
   ```python
   python CollectData2.py
   ```
   After run that file, we will have images that will be data in `Origin` folder (we have already collected 800 images, namely 400/class). Then, run `SplitData.py`
   ```
   python SplitData.py
   ```
   It will split all images from `Origin` folder to `SplitData` folder (with ratio 8/2, we split 638 images for train set and 162 images for test set).

2. Back to `Data` folder and run `PreData`
   ```sh
   python PreData.py
   ```
   This code will call `PreImage.py` (Pre_Image return the landmarks from an image). `PreData.py` will take all images from `SplitData/train` folder and get landmarks from all images, and save them to `PreparedData_train.csv` files. 
   Then, this files will have 638 rows represent for 638 images in train set. Similarly, `PreparedData_test.csv` will contain 162 rows for 162 images in test set.

3. Run `AnalyzeData.ipynb`, return the numpy array data for tranning. 


<!-- CONTRIBUTING -->
## 3. Development History.
<a id="development_history"></a>


![image](https://github.com/DangLeChi/ChiTest/assets/122540817/ba2cf671-7881-4f5b-8c9f-8b70c86406a8)

## 4. How to choose keypoints?
<a id="choose_keypoints"></a>

Firstly, Mediapipe API return 468 landmark points on human face (more 10 points for iris eyes). My team will choose the points that have the biggest change between Normal and Abnormal class. We will find that in `FindAllLandmarks.ipynb` as below:
- Step 1: Take all landmarks points in one image and calculate distances pairwise. 
- Step 2: Calculate mean distance value of all images in one class (Normal/Abnormal), return distance_matrices_normal and distance_matrices_abnormal variable. 
- Step 3: Subtract two variables above to find the different between two classes. 
- Step 4: Find the biggest sum on landmark distances (represent for biggest different), and take the first 20 biggest distance. As a result, we have 17 landmarks and 2 iris center points.  

<!-- USAGE EXAMPLES -->
## 5. Exploratory Data Analysis (EDA).
<a id="exploratory_data_analysis_eda"></a>

### 5.1 Dataframe.
<a id="dataframe"></a>

+ This is the original dataframe.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/e4d25dd3-a569-4571-9b90-00d09bb0dec7)

+ Anomalies of the data set.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/2b644f0b-c7d8-42ae-a412-508080d8fb5a)

+ Delete abnormal values and save dataframe.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/44024f0f-2a57-4b78-81b6-20f965e1b7f5)

### 5.2 Check imbalanced data.
<a id="check_imbalanced"></a>


<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/63439c89-8425-41ae-8d54-092b84d857bd" width="630" height="300">


### 5.3 Kernel Density Plots to visualize values in data.
<a id="kernel_den"></a> 


<!-- ![image](https://github.com/DangLeChi/ChiTest/assets/122540817/a26254e3-b810-457d-9791-a1800014450d) -->
<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/a26254e3-b810-457d-9791-a1800014450d" width="500" height="290">


<!-- ![image](https://github.com/DangLeChi/ChiTest/assets/122540817/e37f89c6-2433-4a82-96bd-48e6519a0afc) -->
<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/e37f89c6-2433-4a82-96bd-48e6519a0afc" width="500" height="290">


### 5.4 PCA.
<a id="pca"></a>

+ Dataframe before PCA.

![image](https://github.com/DangLeChi/ChiTest/assets/122540817/44024f0f-2a57-4b78-81b6-20f965e1b7f5)

+ Dataframe after PCA.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/931668e4-f916-4ab4-ba0d-6ea30cb4689e" width="240" height="360">

+ Histogram for Label in pca_dataframe.

<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/9e86edf7-5dd3-4098-b14b-92589d06c7eb" width="610" height="270">


<img src="https://github.com/DangLeChi/ChiTest/assets/122540817/4541c132-3139-4ce3-9cde-b83dccc7e704" width="610" height="270">

+ Visualize PCA.

![gif_train](SampleImage\visual_pca_train.gif)
![gif_test](SampleImage\visual_pca_test.gif)

### 5.5 Visualize on Image.
<a id="visual_image"></a>

<table>
  <tr>
    <td>
      + Visualize selected keypoints on image.
      
  <br>
      <img src="https://github.com/DangLeChi/ChiTest/assets/122540817/1c317e33-0533-4d29-8a4d-652bda25d4e6" width="450" height="350">
    </td>
    <td>
      + Visual distances between selected keypoints on image.
  
  <br>
      <img src="https://github.com/DangLeChi/ChiTest/assets/122540817/431bb2a9-16c6-464f-b603-5a3bcfcd2c46" width="450" height="350">
    </td>
  </tr>
</table>




<table>
  <tr>
    <td>
      + Visualize selected keypoints in 3D.
      
  <br>
      <img src="https://github.com/DangLeChi/ChiTest/assets/122540817/01540066-b940-4611-819d-3dc39df0b005" width="450" height="350">
    </td>
    <td>
      + Visualize all keypoints on 3D.
  
  <br>
      <img src="https://github.com/DangLeChi/ChiTest/assets/122540817/ec61d54a-4999-43b2-a98e-02a5937afa2b" width="450" height="350">
    </td>
  </tr>
</table>











