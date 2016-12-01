# Predicting Seizures with Long-Term iEEG Recordings
### By Conor B. Murphy

## Summary

While it has long been known that the brain changes state before the onset of a seizure, no reliable clinical application has been developed to forecast seizures until recently.  The world's first clinical trial of this technology is the implantable NeuroVista Seizure Advisory System that employs long-term intracranial electroencephalography (iEEG) to record brain activity linked to drug-resistant, persistent epilepsy.  Forecasting seizures allows patients the opportunity to take fast-acting medications or avoid dangerous activities in addition to reducing anxieties surrounding epileptic events.  This project addresses the most difficult aspect of seizure forecasting by classifying 10-minute recordings as either interictal (baseline) or preictal (prior to seizure) events.  I tested four models finding that gradient boosting and random forest both result in a .91 area under the ROC curve on my validation set.

## Data

The data was provided by a [Kaggle competition](https://www.kaggle.com/c/melbourne-university-seizure-prediction) in collaboration with The University of Melbourne and other sponsors.  The data set consists of 7950 recordings from three patients totaling 40 gb of data.  Each recording is 240k observations of 16 variables, or a 10-minute recording at 400 hz of 16 channels of electrophysiological monitoring.

A growing body of research has identified four stages in the lifecycle of an epileptic event:

1. *Interictal:* a baseline brain state between seizures
2. *Preictal:* the period leading up to a seizure
3. *Ictal:* the seizure itself
4. *Post-ictal:* a period after the seizure

![Lifecycle of an Epileptic Event](https://github.com/conorbmurphy/predicting-seizures/blob/master/figures/epileptic_lifecycle.png)

* Figure 1: The lifecycle of an epileptic event

The most challenging aspect of seizure forecasting is distinguishing between interictal and preictal activity.  The training set is labeled with a 0 for interictal and 1 for preictal, or having a seizure in the period after the recording.  The training set also includes where a given recording falls in a one-hour segment, information that is not available in the test set.

The Kaggle competition suffered from data contamination issues resulting in an extension of the deadline and a significantly truncated test set, which made public leaderboard scores volatile.  There is reason to believe that even after limiting the size of the test set, data contamination issues persisted.  Rather than focus on a high Kaggle score, I redoubled my efforts on a strong model on the uncontaminated data as this creates more actionable insights.

## Exploratory Analysis

Exploratory analysis reveals a few patterns to be explored in greater detail in the feature-building stage.

![Image of an Interictal Recording](https://github.com/conorbmurphy/predicting-seizures/blob/master/figures/interictal.png)

* Figure 2: This plot shows an hour of 16 channels of iEEG data for a period of time not followed by a seizure

![Image of a Preictal Recording](https://github.com/conorbmurphy/predicting-seizures/blob/master/figures/preictal.png)

* Figure 3: This plot shows an hour leading up to a seizure, the seizure itself taking place five minutes after the end of the recording

In the above plots, we can draw attention to some of the general features I focus on in my model.  In recordings not followed by a seizure, we see higher frequency brain activity.  By contrast, the preictal recording shows lower frequency activity.

## Feature Building

I built features surrounding a variety of hypotheses, each will be explored in detail below.  The features were in five basic categories creating a total of 819 features for each recording in the final model:

* 160 channel means from 1-minute segments
* 400 wavelet transformations
* 118 method of moments calculations
* 16 entropy calculations
* 122 pearson correlations
* 3 patient number dummies

The final feature importances are as follows:

![Feature Importance](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/feature_importance.png)

* Figure 4: A feature importance plot shows how the wavelet transformations contributed the most to my final model

### Wavelet Transformation

Given that the frequency of brain electrical activity appears to correlate with whether a seizure is immanent, I performed a wavelet transformation on the data with five transformations within the bounds of each of the common wavelengths:

| Wavelength | Frequency (hz)        |
| ---------- |:---------------------:|
| delta      |     < 4               |
| theta      |     >= 4 hz & < 8 hz  |
| alpha      |    >= 8 hz & < 14 hz  |
| beta       |    >= 14 hz & < 32 hz |
| gamma      |    >= 14 hz           |

A wavelet spectrogram demonstrates which wavelengths are active for interictal and preictal recordings.  A 3 hz 'spike-and-wave' wavelet is common in preictal recordings, as is other lower freqency activity.  3 hz would appear as 133 hz in the spectrograms below.

![Interictal Wavelet Spectrogram](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/spectrogram_i.png)

* Figure 5: The interictal wavelet spectrogram shows the prevalence of higher frequency activity (modeled on the y axis)

![Preictal Wavelet Spectrogram](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/spectrogram_p.png)

* Figure 6: The preictal spectrogram shows the absence of higher frequency activity

### Pearson Channel Correlation

Channel correlations contributed significantly to the final model as well.  The recordings were pre-processed to take into account the difference between a given trace and surrounding traces.  Despite this, I was able to pull out insights from different channels using Pearson correlation.  Plotting the kernel density estimates from these correlations shows similar probabilities, indicating that the model is picking up on the interaction of features over the values of the correlations themselves.   An example of higher positive or negative correlations can be seen in the plots below.

![Interictal Channel Correlations](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/coorelations_i.png)

* Figure 7: A heatmap of the interictal channel correlations shows higher positive and negative values

![Preictal Channel Correlations](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/coorelations_p.png)

* Figure 8: A heatmap of the preictal correlations shows less correlated channels

### Entropy

Shannon entropy offers an assessment of irregularity in the EEG recordings.  A kernel density estimation estimates the probability distribution of seeing given observations in each channel for each recording.  With this probability distribution, I computed this measure of irregularity, resulting in a slight gain in my model's performance.

### Method of Moments and Channel Means

In addition to the above, I calculated variations on the method of moments including the following:

1. Channel mean for the entire recording and in 1-minute segments
2. Variance (channel and total)
3. Channel skew
4. Channel kurtosis
5. Channel and total minimum, maximums and medians

## Modeling

After building the features described above, I built a model that took into account unbalanced classes (around 90% of the data is interictal), missing data, encoding the categorical variable for patient number and normalization.

I tried the following techniques:

1. *Logistic Regression:* using L1 regularization due to high dimensionality of data set
2. *Random Forest:* with grid search to tune parameters
3. *Gradient Boosting:* using the XGBoost implementation with nominal adjustments
4. *SVM:* Using both Radial Basis Function (RBF) and linear kernels.  I tried a linear kernel to reduce overfitting however the RBF still outperformed it on the training and validation sets.

![ROC Curve](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/roc_curve.png)

* Figure 9: The ROC curves for each model shows the true positive and false positive rates for different decision thresholds

The scoring metric I decided to use with area under the ROC curve as this matched the Kaggle competition.  Scores were calculated using 5-fold cross-validation on the training set and a prediction on a withheld validation set.  All analyses were done on what Kaggle was referring to as the training set since this data was labeled.  Scores on the cross-validated and validation sets are included below with the format (cross validated score / validation score):

| Patient    | Logistic Regression | Random Forest | XGBoost^ | SVM          |
| ---------- |:-------------------:|:-------------:|:--------:|:------------:|
| Combined   | 0.81 / 0.88         | 0.88 / 0.91   | 0.91     | 0.84 / 0.87  |
| A          | 0.84 / 0.85         | 0.85 / 0.90   | 0.90     | 0.84 / 0.87  |
| B          | 0.90 / 0.88         | 0.85 / 0.86   | 0.88     | 0.87 / 0.90  |
| C          | 0.86 / 0.85         | 0.91 / 0.93   | 0.93     | 0.87 / 0.87  |

^ XGBoost was not cross-validated

## Reproducing my Analysis

The feature building, modeling and visualization can all be recreated using the code in this repository.  I completed my work on a 40-core AWS EC2 instance to take full advantage of parallelizing the computationally demanding parts of this analysis (especially the wavelet transformations).

The code in `feature_building.py` translates files from the root directory `/data` into six csv files: one training and one test set for each of the three patients.  It parallelizes the operation across 40 cores and saves the result in the root project folder.  To recreate this analysis, the number of cores in the function `reduce_parallel()` can be changed to match your requirements and the hard-coded directories in that function can be changed as well.  Since this analysis takes a few hours, I saved the consolidated files to the data directory, divided by patient to make sure the computation did not run out of RAM.

The models can be run using `model.py`, which will print out the scores seen in the table above.  It also includes functions for using grid search on a few of the models as well as ROC curves.  The code in the final code block will save feature importances and probability predictions on the validation set for plots needed in `visualizations.py`

Finally, the figures present in this file can be recreated by running `visualizations.py`.  

## Next steps

The models that took into account feature interaction such as random forest and gradient boosting outperformed logistic regression substantially.  Since logistic regression does not account for feature interaction, there is likely missed, non-linear relations that accounts for this difference in scores.  In future models, this can be accounted for with interaction terms or with dimensionality reduction using non-negative matrix factorization (which has been shown to account for correlations better than principal component analysis).

Considering that the data set only includes base iEEG recordings, there are a few steps that could create a more accurate prediction by using side data.

1. A metric for the severity of a patient's epilepsy could create a more or less sensitive alert threshold.
2. A calibration protocol could allow the users with similar brain activity to be clustered.  For instance, this could be accomplished by asking them to perform certain mental tasks in order to get a better idea for the range of their baseline, normal brain activity.
3. Using activity data such as motion and body position from the recording device could better classify the cause of a given brain state.

In addition to the models I tried, convolutional neural nets have been shown to be effective with this type of data.  Experimenting with other wavelets such as the Morlet and the 3hz 'spike-and-wave' common in preictal recordings could also yield different results in the wavelet transformation.  A draft function to be used Scipy's `cwt()` function can be called with `from code.model import morlet`

## Acknowledgements

I undertook this project as part of my capstone project as a Data Science Fellow at Galvanize in San Francisco.  While I found all of the instructors to be deeply knowledgable and helpful, I wanted to thank Moses in particular for all his help.  I made a game out of trying to stump him with obscure questions dug out from the depths of various mathematician's wikipedia pages and was ultimately unsuccessful.  Thanks Moses!
