                      Braille Conversion of Script to Audio

1) Datasets 
    - https://www.kaggle.com/shanks0465/braille-character-dataset
    - Second dataset is collected manually and done some augumentation to increase the data.
2) DeepLearning Model Used
    - CNN
3) Accuracy
    - 91%
4) Custom input Preprocessing
    - Input Reading i.e, Image aquisation
    - Image Preprocessing
    - Image Segmentation
    - Text from image
        - Passing every segment to our trained model to predict corresponding character
        - Concatenation all the characters finally.
    - Text to Speech
        - using gTTS(Google Text To Speech)
