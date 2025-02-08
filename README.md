# Music Genre Classification using CNN

This project uses Convolutional Neural Networks (CNN) to classify music genres from audio files.

## Requirements

- Python 3.6+
- TensorFlow
- Keras
- NumPy
- Librosa
- Matplotlib

You can install the required packages using:
```bash
pip install tensorflow keras numpy librosa matplotlib
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Music-Genre-Classification-using-CNN.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Music-Genre-Classification-using-CNN
    ```
3. Run the main script:
    ```bash
    python main.py
    ```

## main.py

The `main.py` script performs the following tasks:
1. Loads and preprocesses the audio data.
2. Builds and compiles the CNN model.
3. Trains the model on the dataset.
4. Evaluates the model's performance.

## Dataset

The dataset used for training and evaluation should be organized in the following structure:
```
dataset/
    genre1/
        file1.wav
        file2.wav
        ...
    genre2/
        file1.wav
        file2.wav
        ...
    ...
```

## Results

After training, the model's performance metrics and loss/accuracy plots will be displayed.

## Acknowledgements

- [Librosa](https://librosa.org/) for audio processing.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for building and training the neural network.

## Model Architecture
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 11, 128, 32)       320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 5, 64, 32)        0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 5, 64, 32)        128       
 ormalization)                                                   
                                                                 
 dropout (Dropout)           (None, 5, 64, 32)         0         
                                                                 
 conv2d_1 (Conv2D)           (None, 3, 62, 64)         18496     
                                                                 
 ... (full architecture in notebook)
=================================================================