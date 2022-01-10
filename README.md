# NLP-Sentiment-Analysis

This is a project that analyzes sentiment in movie reviwes using NLP. 

# Motivations

I am completing a larger project on emotional sentiment analysis and wanted to gain some experience with NLP and ML before.

# Challenges

The main challenges were finding ways to tweak models and waiting long for the code to run. I tried to also include the SVM and Nu-SVM but I wasn't able to make them converge.

# Notes

One of the pickle files, feature_sets.pickle was not abel to be uploaded because it exceed a size of 100mb. If you want to run the model it should only be a few seconds to run that portion of the code and generate the file.
I'm trying to find a way around the hard 100mb limit but there doesn't seem a good way.

# Next Steps

The majority of the Twitter streaming code is written, I just need to gain approval from Twitter devs to stream tweets from Twitter.

# Accuracy Updates

The most recent test I did the model was 81.6% accurate before throwing out any data that had a confidence level of under 80%.
