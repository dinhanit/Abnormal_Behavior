# Exam Abnormal Behavior Recognition (Data Branch)

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

![Image](Readme\road.png)


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started
<a id="getting-started"></a>

This is an instructions on setting up project locally.
### Pipeline
<a id="Pipeline"></a>

This is our flow to run this branch
![Image](Readme\pipeline.png)

### Installation
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

### Instructions Manual
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Exploratory Data Analysis (EDA)
<a id="exploratory_data_analysis_eda"></a>

### Dataframe.
<a id="dataframe"></a>

+ This is the original dataframe.

![original_dataframe](Readme\dataframe_old.png)

+ Anomalies of the data set.

![abnormal_data](Readme\abnormal_data.png)

+ Delete abnormal values and save dataframe

![dataframe](Readme\dataframe.png)

### Check imbalanced data.
<a id="check_imbalanced"></a>


![check_imbalanced](Readme\check_class.png)

### Kernel Density Plots to visualize values in data.
<a id="kernel_den"></a> 


![kernel_density_train](Readme\kernel_density_train.png)
![kernel_density_test](Readme\kernel_density_test.png)

### PCA.
<a id="pca"></a>


![pca](Readme\pca.png).



+ Histogram for Label in pca_dataframe.
![histogram_train](Readme\histogram_train.png)
![histogram_test](Readme\histogram_test.png)

+ Visualize PCA.
![gif_train](Readme\video_train.gif)
![gif_test](Readme\video_test.gif)

### Visualize on Image.
<a id="visual_image"></a>

+ Visual keypoints on image.

![visual_image](Readme\visual_image.png)

+ Visual distance keypoints on image.

![visual_distance](Readme\visual_distance.png)

+ Visual line image on 2D.

![visual_2d](Readme\visual_2d.png)

+ Visual keypoints on 3D.

![visual_3d](Readme\visual_3d.png)

+ Visual all keypoint on 3D.
![visual_all_landmark](Readme\visual_all_landmark.png)


<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
## Development History.
<a id="development_history"></a>

![history](Readme\history.png)
<p align="right">(<a href="#readme-top">back to top</a>)</p>






