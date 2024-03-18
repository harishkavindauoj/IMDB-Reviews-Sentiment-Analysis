# IMDB Reviews Sentiment Analysis with PyTorch and Flask

This project utilizes a Convolutional Neural Network (CNN) model trained on the IMDB dataset to analyze sentiment in movie reviews. The trained model is deployed using Flask for real-time web-based sentiment analysis.

## Overview
The project leverages PyTorch for training a CNN model to classify movie reviews into positive or negative sentiments. It also employs Flask to deploy the trained model as a web application, enabling users to input movie reviews and receive sentiment predictions instantly.

## Usage
1. **Training the Model**: Execute `model.py` to train the CNN model on the IMDB dataset using PyTorch.
2. **Running the Web Application**: Start the Flask server by running `app.py`. Access the sentiment analysis interface at `http://localhost:5000` in your browser.

## Dependencies
- Python 3.x
- PyTorch
- Flask
- NLTK
- pandas

## Credits
- IMDB Dataset: [Keggle] (https://www.imdb.com/](https://www.kaggle.com/datasets/bhavikjikadara/imdb-dataset-sentiment-analysis))

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.
