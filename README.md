# Forecasting Seizures with Long-Term iEEG Recordings
### By Conor B. Murphy

## Summary

While it has long been known that the brain changes state before the onset of a seizure, no reliable clinical application has been developed to forecast seizures until recently.  The world's first clinical trial of this technology is the implantable NeuroVista Seizure Advisory System that employs long-term intracranial electroencephalography (iEEG) to record brain activity linked to drug-resistant, persistent epilepsy.  Forecasting seizures allows patients the opportunity to take fast-acting medications or avoid dangerous activities in addition to reducing the anxieties surrounding epileptic events.  This project addresses the most difficult aspect of seizure forecasting by classifying 10-minute recordings as either interictal (baseline) or preictal (prior to seizure) events.  I found that an ensemble method using logistic regression, gradient boosting, random forest and support vector machines gave me the best results of a .82 area under the ROC curve.

## Data

The data was provided by a [Kaggle competition](https://www.kaggle.com/c/melbourne-university-seizure-prediction) in collaboration with The University of Melbourne and other sponsors.  The data set consists of 7950 recordings from three patients totaling 40 gb of data.  Each recording is 240k observations of 16 variables, or a 10-minute recording at 400 hz of 16 channels of electrophysiological monitoring.

A growing body of research has identified four stages in the lifecycle of an epileptic event:

1. *Interictal:* a baseline brain state between seizures
2. *Preictal:* the period leading up to a seizure
3. *Ictal:* the seizure itself
4. *Post-ictal:* a period after the seizure

The most challenging aspect of seizure forecasting is distinguishing between interictal and preictal activity.  The training set is labeled with a 0 for interictal and 1 for preictal, or having a seizure in the next five minutes.  The training set also includes where it falls in a one-hour segment, information that is not available in the test set.

The Kaggle competition suffered from data contamination issues resulting in an extension of the deadline and a significantly truncated test set, which made public leaderboard scores volatile.  There is reason to believe that even after limiting the size of the test set, data contamination issues persisted.  Rather than focus on a high Kaggle score, I redoubled my efforts on a strong model on the uncontaminated data as this likely better results in more actionable insights.

## Exploratory Analysis

Exploratory analysis reveals a few patterns to be explored in greater detail in the feature-building stage.

![Image of an Interictal Recording](https://github.com/conorbmurphy/predicting-seizures/blob/master/figures/interictal.png)

* Figure 1: This plot shows an hour of 16 channels of iEEG data for a period of time not followed by a seizure

![Image of a Preictal Recording](https://github.com/conorbmurphy/predicting-seizures/blob/master/figures/preictal.png)

* Figure 2: This plot shows an hour leading up to a seizure, the seizure itself taking place five minutes after the end of the recording

In the above plots, we can draw attention to some of the general features I focus on in my model.  In recordings not followed by a seizure, we see higher frequency brain activity.  By contrast, the preictal recording shows lower frequency activity.

## Feature Building

I built features surrounding a variety of hypotheses, each will be explored in detail below.  The features were in five basic categories creating a total of 819 features for the final model:

* 160 channel means
* 400 wavelet transformations
* 118 method of moments calculations
* 16 entropy calculations
* 122 correlations
* 3 patient number dummies

The final importances are as follows:

![Feature Importance](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/feature_importance.png)

* Figure 3: Importances of the categories of features.

### Wavelet Transformation

Given that the frequency of brain electrical activity appears to correlate with whether a seizure is immanent, I performed a wavelet transformation on data with five transformations within the bounds of each of the common wavelengths:

| Wavelength        |      Frequency (hz)      |
| ------------- |:-------------:|
| delta      |     < 4     |
| theta      |     >= 4 hz & < 8 hz   |
| alpha |    >= 8 hz & < 14 hz   |
| beta |    >= 14 hz & < 32 hz   |
| gamma |    >= 14 hz   |

A wavelet spectrogram demonstrates which wavelengths are active for interictal and preictal recordings.

![Interictal Wavelet Spectrogram](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/spectrogram_i.png)

![Preictal Wavelet Spectrogram](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/spectrogram_p.png)

### Pearson Channel Correlation

![Interictal Channel Correlations](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/coorelations_i.png)

![Preictal Channel Correlations](https://github.com/conorbmurphy/Predicting-Seizures/blob/master/figures/coorelations_p.png)

### Entropy

Shannon entropy offers an assessment of irregularity in the EEG recordings.  A kernel density estimation estimates the probability distribution of seeing given observations in each channel for each recording.  With this estimation, I computed this measure of irregularity.

VISUALIZE ENTROPY

### Method of Moments

In addition to the above, I calculated variations on the method of moments including the following:

1. Channel mean for the entire recording and in 1-minute segments
2. Variance (channel and total)
3. Channel skew
4. Channel kurtosis
5. Channel and total minimum, maximums and medians


## Modeling

After building the features described above, I built a model that took int account unbalanced classes (around 90% of the data is interictal), missing data and normalization.

Modeling on just the last segment-how did that work?
Techniques tried - RF, SVM, LR, XGB
Dimensionality reduction
Imbalanced classes
ROC curves

The scoring metric I decided to use with area under the ROC curve as this matched the Kaggle competition.  Scores were calculated using 5-fold cross-validation on the training set and a prediction on a withheld validation set.  Both results are included below:

| Patient    | Logistic Regression | Random Forest | XGBoost | SVM   |
| ---------- |:-------------------:|:-------------:|:-------:|:-----:|
| Combined   | 0.84                | 0.88          | 0.87    | 0.84  |
| A          | 0.80                | 0.86          | 0.90    | 0.85  |
| B          | 0.84                | 0.83          | 0.83    | 0.82  |
| C          | 0.86                | 0.91          | 0.94    | 0.87  |


## Reproducing my Analysis

I completed my work on a 40-core AWS EC2 instance to take full advantage of parallelizing the computationally demanding parts of this analysis.  The code in `feature_building.py` translates files from the root directory `/data` due to the size of the data set.  It parallelizes the operation across 40 cores and saves the result in the root project folder.  To recreate this analysis, the number of cores in the function `reduce_parallel()` can be changed to match your requirements and the hard-coded directories in that function can be changed as well.

Since this analysis takes a few hours, I saved the consolidated files to the data directory, divided by patient to make sure the computation did not run out of RAM.

The model can be run using `model.py`

The figures present in this file can be recreated by running `visualizations.py`

## Next steps

Considering that the data set only includes base iEEG recordings, there are a few steps that could create a more accurate prediction by using side data.

1. A metric for the severity of a patient's epilepsy could create a more or less sensitive alert threshold.
2. A callibration protocol could allow the users with similar brain activity to be clustered.  For instance, this could be accomplished by asking them to perform certain mental tasks in order to get a better idea for what the range of their baseline, normal brain activity.
3. Using activity data such as motion and body position from the recording device could better classify the cause of a given brain state.

In addition to the models I tried, convolutional neural nets have been shown to be effective with this type of data.  Experimenting with other wavelets such as the Morlet could also yeild different results in the wavelet transformation.  A draft function to be used Scipy's `cwt()` function can be called with `from code.model import morlet`



## Acknowledgements

## References
