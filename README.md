# MachineLearning Project for Image Classfication in Python 
**Coded in Python using Google Colab**

This project aims to classify images to help enforce COVID Mask Rules.

**Introduction**: The mask-mandate for protection against the COVID-19 pandemic has affected the field of facial recognition, where a need has risen to recognise faces with masks on. Efficient recognition systems are expected to check that faces are masked in regulated areas. This project focuses on the facial recognition used to check if the detected face is correctly masked or unmasked, which can help with the enforcing the mask mandate. As an added level of complexity, this coursework looks at detecting incorrectly placed masks as well. Many people do not wear the masks correctly simply due to bad practice, lack of understanding or due to vulnerabilities of individuals. The class labels for this classification task have been chosen as:

- Unmasked

- CorrectlyMasked

- IncorrectlyMasked


## Data Compilation 
Keeping the three class labels in mind, a search was carried out for suitable images.

The Unmasked dataset was created from the Flickr-Faces-HQ(FFHQ) which contains high quality images of human faces and was created as a benchmark for generative adversarial networks (GAN).This dataset contains a good variation in terms of age, ethnicity and also variation of accessories like hats and glasses which the model can learn from. The Masked and Incorrectly masked data was compiled from MaskedFace-Net contains images of human faces with a correctly or incorrectly worn mask (133,783 images) based on the FFHQ dataset mentioned above. Masks have been artificially applied to the FFHQ dataset according to the categories.

Incorrectly Masked contains 3 sub-categories:

- Uncovered nose *chosen as IncorrectlyMasked
  
- Uncovered chin
  
- Uncovered nose and mouth.
  
For the purposes of this coursework the granularity of the of the dataset is not exploited to the fullest extent, as the Incorrectly Masked currently contains only 'Uncovered nose' images, as it is the mostly commonly found in real life.This is because this project is considered as a first step in experimentation and not a final solution.

The incorrectly labelled and unclear images were removed. The masked and incorrectly masked images of children under 3 were removed as well, as it is an impractical scenario.Then 2000 of each of the three categories was chosen for a range of ethnicity,ages and accessories.


## Sample Images

![image](https://github.com/Surya-LR/MachineLearning_ImageClassfication_Python/assets/77691667/3006f53b-5837-4975-824a-d62d947bdb23)


**Final Selection**: The data was imbalanced to simulate real conditions, through code. The sizes now are:

- Unmasked: 1000

- CorrectlyMasked: 800

- IncorrectlyMasked: 100

## The combination of preprocessing and classifiers

The Layout of the experiments in terms of the processing and classifiers used is shown in the image below:

![image](https://github.com/Surya-LR/MachineLearning_ImageClassfication_Python/assets/77691667/2e08f159-4ac5-4990-b62b-583cf7425de1)


Layout of the two submitted files are shown below:

![image](https://github.com/Surya-LR/MachineLearning_ImageClassfication_Python/assets/77691667/ba248c7c-5429-47ae-b15f-6f6093a4c064)

  
## The training and testing framework for the classfiers

![image](https://github.com/Surya-LR/MachineLearning_ImageClassfication_Python/assets/77691667/f7c358a5-2f17-4d70-b280-d784ad67372c)

## Reflection on Results:

For the original repo, best results for ROC curve can be seen from Random Forest.It also has the best score for precision and recall and f1-score.The AdaBoost and SVC shows good results,but the recall for the minority class (IncorrectlyMasked), showing worser results as expected. This can be confirmed with the confusion matrix as well.

The pre-processed repo shows slightly worser results compared to original repo. RF still performs the best overall according to ROC curve and the precison, but Adaboost has better recall for the minority class.

It is seen that the pre-processing steps applied did not help improve the classification results.Alternate pre-processing steps should be considered, as the recall for the minority class has dropped significantly.

The feature extraction(FE) repo with HOG and PCA shows the best results with SVM, which is in agreement with the research paper mentioned in the FE section.The recall for the minority class shows improvement as well.But the overall scores are better with original repo.

Definite improvement in the ROC curves can be see with the augmented repo for all three classifiers with the scores of Adaboost and RF quite close for precision recall and f1-score, with the Adaboost getting overall better scores due to the precision in CorrectlyMasked.

CDSMOTE shows good results for ROC curve with RF showing better results than the other two classifiers in all the metrics. But the overall results are not better than original or augmented repo for any of the metrics.

For CNN early stopping and call back was implemented. With the confusion matrix, best results are shown with the original repo and augmented repo.With the Pre-processed repo and CDSMOTE showing high recall for the minority class, but poor precision.

