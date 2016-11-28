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

![KDE Plot](https://github.com/conorbmurphy/predicting-seizures/blob/master/figures/kde2.png)

* Figure 3: In the above kernel density estimator plot we can see that most channels have higher frequency activity in non-seizure activity compared to pre-seizure brain states

### Wavelet Transformation

Given that the frequency of brain electrical activity appears to correlate with whether a seizure is immanent, I performed a wavelet transformation on data with five transformations for each of the common wavelengths:

| Wavelength        |      Frequency (hz)      |
| ------------- |:-------------:|
| delta      |     < 4     |
| theta      |     >= 4 hz & < 8 hz   |
| alpha |    >= 8 hz & < 14 hz   |
| beta |    >= 14 hz & < 32 hz   |
| gamma |    >= 14 hz   |



### Channel Correlation

Make heatmap of correlations

### Other Features?  MOM?  Entropy

## Feature Building

List of features
Why
What worked, what didn't
Development environment

## Modeling

Modeling on just the last segment-how did that work?
Techniques tried - RF, SVM, LR, XGB
Dimensionality reduction
Imbalanced classes
ROC curves

## Key Findings

## Next steps

## Acknowledgements

## References
