
## Tensorflow World 2019
#### Zero to ML Hero with TensorFlow 2.0
*by [Laurence Moroney](https://github.com/lmoroney) (Google)*

![Tensorflow World Coding](/tfw19/imgs/tfworld_coding.png)

**09:00 AM: Keras & TF Introduction**
Google Colab for development notebooks ad run environment
- Github [Link](https://github.com/lmoroney/mlday-tokyo)
- Google Colab [Link](https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab1-Hello-ML-World.ipynb#scrollTo=fA93WUy1zzWf) 
- Colab can also work on local machine using a proper version of JupyterLab


[LAB 1](https://bit.ly/tfw-lab1)
- Try: Q:https://bit.ly/tfw-ex1q and A:https://bit.ly/tfw-ex1a

- Note: Time taken per epoch ay increase over time since the floating point values may become 
- Idea: Normalize all values between 0-1 since it keeps computations within limits

[LAB 2]()
- Try: Q:https://bit.ly/tfw-lab2cv and A:https://bit.ly/tfw-lab2cva
- Matching raw pixels to output

**10:30 AM: Image, Filters & CNN**

[LAB 3](https://bit.ly/convolutions-fun)

- CNN: Underlying concepts that requires us to find the filters that identifies the features that we are interested in.
- tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1))
    - This layer generates 64 random Conv filters of size 3x3
    - Image shape will be reduced from 28x28x1 to 26x26x64. 
    - In 26x26x64, first two are the image dimensions being reduced and last dimension is the filters
- model.summary() shows the networks layer/shape details 

[LAB 4](https://bit.ly/tfw-lab4)
- Using convolutions
- Try: Q:https://bit.ly/tfw-lab4exq and A:https://bit.ly/tfw-lab4exa

**12:00 PM: Complex CNN Networks**

[LAB 5]()

- Dataset: Generated CGI horse & human images. Works well with real images except on Elon Musk
- Keras trick: Put images in a label-named-directory and ImageDataGenerator will take care of attaching that label to the image
- Sometimes we provide strings for loss or optimizer and sometimes an object to which we can pass params
- *Not covering Gradient Descent.* Refer to [Andrew Ng on Youtube](https://www.youtube.com/watch?v=uJryes5Vk1o)
- Data quality is the most critical aspect
- Upload files to colab and run through the test predictions

**01:00 PM: Post Lunch Horse classification**

[LAB 6](https://bit.ly/tfw-lab6)
- Try: Q: https://bit.ly/happy-or-sad-q and A: https://bit.ly/tfw-happyorsad-a

**02:35 PM: Features & Overfitting**
- Data quality is the most critical aspect
- Data should typically be balanced to reduce bias in the dataset. 
- Overfitting (Solutions)
    - Image augmentation. (*No rule as to how much augmentation we should be using !!!*)
        - ImageDataGenerator object takes the rotation_range, shift, skew, flip and zoom etc.
            - Some cases, rotation by 180 worked better and val/train accuracy narrowed
            - No callback on what augmentations happened and is purely random. No way to report augmentation
        - Subclass the generator and write a custom augmentation
        - Try https://bit.ly/hh-augmented and https://bit.ly/cd-augmented 

**03:30 PM: NLP Introduction**
- Tokenizing (Encoding sentences to Tokens)
    - "Listen" and its anagram "Silent"
    - Sentence and assessing similarity in sentence

[NLP 1](https://bit.ly/tfw-nlp1)

- Code
    - from tensorflow.keras.preprocessing.text import Tokenizer
        - Understands the diff between capitalization and symbols 
        - Specify Out-Of-Vocabulary "OOV_TOKEN" so that meaning of sentenses in a corpus is not missing
        - [Padding/Masking to sequences](https://www.tensorflow.org/guide/keras/masking_and_padding) which will simplify shape. There are ragged sequences which can train without padding.
            - Excess padding will impact losses in training. Better truncate the long one to short.
            - Keras has masking functions for scenarios like that

[NLP 2](https://bit.ly/tfw-nlp2)

- Training 
    - Train, Test split by subsetting the numpy array 
    - Understand/Establish numerical sentiment in a word or sentence numerically by embedding.
        - Bad[-1,0], Meh[-0.4,0.7], NotBad[0.5,0.7], Good[1,0] 
        - Think in a cartesian coord system, Good and bad are opposite in meaning while Meh is slightly negative.
        - A general estimate is 16 dimensions embdding for 10's thousands of words 
- Explore [Sarcam in news headlines dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection) by Rishabh Misra
    - [NLP 3](https://bit.ly/tfw-nlpsarcasm) 
        - Sarcasm Data Notebook
    - Data has post-padding & shape (26709, 40)
- Explore [BBC News Dataset](https://www.kaggle.com/shineucc/bbc-news-dataset)
    - [NLP 4](https://bit.ly/tfw-nlpbbc) 
        - BBC Data Notebook. Stop words removed before tokenization
- [NLP 5](https://bit.ly/tfw-sarcembed)
    - Embedding Projector http://projector.tensorflow.org, 
    - Upload/Load vect.tsv, meta.tsv 
    - Select sphereize data to visualize the PCA 
- [NLP 6](https://bit.ly/tfw-bbc) 
    - "Some bug exists here since it classifies everything as Business"
        


**Alternate Sneak peaks**
- ML in production: Getting started with [TensorFlow Extended(TFX)](https://medium.com/tensorflow/what-exactly-is-this-tfx-thing-1ac9e56531c) *by Robert Crowe (Google)*
    - Apache Airflow (DAG) based pipelines
    - Described in [KDD 2017 paper](https://www.kdd.org/kdd2017/papers/view/tfx-a-tensorflow-based-production-scale-machine-learning-platform)
- Hyperparameters tuning for TensorFlow using Katib and KubeFlow *by Neelima Mukiri and Meenakshi Kaushik(Cisco)* 
    - Tutorial https://tfworldkatib.github.io/tutorial
    - AutoML using Katib [Link](https://tfworldkatib.github.io/tutorial/katib/katib.html)
