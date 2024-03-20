# Visual Speech Recognition using LipNet
Visual speech recognition using deep learning involves the utilization of neural network architectures to predict speech content solely based on visual cues extracted from videos of speakers. This technology holds significant potential for various applications, including improving speech recognition accuracy in noisy environments, aiding individuals with hearing impairments, enhancing human-computer interaction in multimedia systems, and facilitating automatic transcription of videos.

## LipNet Architecture

![LipNet](https://github.com/TejasPatne/visual-speech-recognition/assets/107361404/39dafc3f-77e7-4fcb-9510-4b86df56c9f4)

**Overview:**
LipNet is a deep learning model designed for visual speech recognition. Recent research has proposed replacing the traditional CNN layers with Spatio-Temporal Convolutional Neural Networks (STCNN), typically implemented using 3D Convolutional Neural Networks (3DCNN). These modifications aim to enhance the model's ability to extract both spatial and temporal features from video inputs.

## Dataset

**Description:**
To facilitate experimentation and training, I've included the `gridcorpus.sh` script file. Running this script in your terminal will automatically download the Grid Corpus dataset, a popular dataset for visual speech recognition tasks.

## Getting Started
  1. **Clone the Repository**: Clone this repository to your local machine using the following command:
     ```
     git clone [https://github.com/your_username/your_repository.git](https://github.com/TejasPatne/visual-speech-recognition.git)
     ```
  2. **Download Dataset**: Run the `gridcorpus.sh` script in your terminal to download the Grid Corpus dataset:
     ```
     sh gridcorpus.sh
     ```
  3. **Explore the Notebook**: Open and explore the `lipnet.ipynb` notebook to understand the LipNet implementation and experiment with training the model.
  4. **Modify and Experiment**: Feel free to modify the notebook, experiment with different hyperparameters, or even try different architectures to further enhance the performance of the visual speech recognition model.
  
## Contributions

**How to Contribute:**
Contributions to this project are welcome! Whether it's bug fixes, feature enhancements, or additional documentation, your contributions can help improve the project for everyone. Feel free to fork the repository, make your changes, and submit a pull request.

Thank you for visiting this repository and showing interest in my final year major project on Visual Speech Recognition using LipNet. If you have any questions or suggestions, feel free to reach out!
