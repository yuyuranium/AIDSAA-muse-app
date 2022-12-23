# Music Emotion Classification

This project aims to perform classification on music emotion based on the *NVIDIA* Jetson Xavier NX board.

## Data Preprocess

![Data Preprocess](./img/preprocess.png)

* Shuffle all the dataset to make training and testing set have randomly selected data
* Split the training and testing dataset
* Training set: 90% of all the data
* Testing set: 10% of all the data
* Slicing original music data to make each data smaller and make the classes balance
  * Number of tense and sad musics are about three times of the joy and peace musics
* Sliding window = 2s
* For tense and sad classes: stride = 1s
* For joy and peace classes: stride = 0.35s
* Add noises to make the model more robust and simulate the microphone recording situation
  * Randomly generated white noise
* Aside from the music data, we also calculated the high level features as another input of the model
  * `mfcc`
  * `rolloff`
  * `spectral_contrast`
  * `rms`

## Model

### Model Input & Output

* **Input**:
  1. Time series audio data: array of shape (44100, 1)
  2. Feature data derived from the music segment in 2 seconds
* **Output**
  * Classification result in one-hot encoding

### Structure

![Model Structure](./img/model_structure.png)

### Training Curve

![Training Curve](./img/training_curve.png)

## Analysis

### Evaluation

* **Test loss**: 2.073
* **Test accuracy**: 0.642

### Confusion Matrix

![Confusion Matrix](./img/confusion_matrix.png)

## GUI Application

### User interface

* We provide a simple and easy-to-use graphical user interface in our application
  * One preview window for the recorded audio
  * One text display to show the information
  * One start/stop button to control the app

![GUI](./img/gui.png)

### Functionality

* Pipelined recording and predicting

![img](https://lh6.googleusercontent.com/jOCGfYhHmL_aB_yQR_Ovy4nLlv65903_MNCBoePMLqIeexJY243e59Bp2JSeqxSvmEb0v3hkmOV3jO_YuNSYTQTIAx5SLgdyBcrkwMJ2wzLACkFmsOiYR6O42_1LC9UZKLd4ml3G3lHDCDDYLyYo-_TPSw)

* No need to manually stop recording and start inferencing
* Exploit GPU on Xavier to accelerate model inference
  * On average the model can finish inference within 90ms

![img](https://lh5.googleusercontent.com/eI0HsY_EkhTEEUnCfA_F5FwzNXh6QJqPdnilqu-OKHMBif0TCIcBeAkfLH-PjC8U2mnfl9q3H9-KpVtqn_9lmga_WLcQrJjhqDq9wdQ-YSc5XlV3ohPRF9Il5NL6-Pl-gkSx-6AS5NtCmXeAtZK8RkmJmg)

## Work Division

| Name         | Work            |
| ------------ | --------------- |
| `0x26rew`    | Training        |
| `choucl`     | Preprocessing   |
| `yuyuranium` | GUI development |
| `jumha`      | Training        |
