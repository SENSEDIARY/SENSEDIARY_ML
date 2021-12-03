# Sense_Diary_ML

## Sentiment Diary ML Repository

### Data

* #### Train Data 
  * [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140)
* #### Content Data
  * [YouTube Trending Video Dataset](https://www.kaggle.com/rsrishav/youtube-trending-video-dataset)
  * [Trending YouTube Video Statistics](https://www.kaggle.com/datasnaek/youtube-new)

### Model
* #### Emotion labeling model
  * [torchMoji](https://github.com/huggingface/torchMoji)
* #### Sentiment classification model
  * [ktrain](https://github.com/amaiya/ktrain)

### Environment
* [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ko)

### Code
* [sentiment140_preprocess](sentiment140_preprocess.ipynb) preprocess train data 
* [sentiment140_torchmoji_label](sentiment140_torchmoji_label.ipynb) sentiment labeling with torchMoji (using [texts_sentiment_label.py](/torchMoji/examples/texts_sentiment_label.py) file)
* [sentiment140_torchmoji_sentiment_preprocess](sentiment140_torchmoji_sentiment_preprocess.ipynb) preprocess sentiment label (reduce label 64 to 3)
* [ktrain_sentiment_multiclassification](ktrain_sentiment_multiclassification.ipynb) learn and save ktrain sentiment classification model
* [unable_youtube_remove](unable_youtube_remove.ipynb) remove unable youtube contents in csv
* [youtube_preprocessing](youtube_preprocessing.ipynb) Pre-processing YouTube data to classify emotions (US English)
* [youtube_preprocessing_kr](youtube_preprocessing_kr.ipynb) Pre-processing YouTube data to classify emotions (Korean)
* [youtube_sentiment_label](youtube_sentiment_label.ipynb) youtube sentiment label predict with ktrain 
