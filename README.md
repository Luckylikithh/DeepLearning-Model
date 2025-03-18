# DeepLearning-Model

# Spotify Song Popularity Prediction

## Overview
This project aims to predict the popularity of songs on Spotify using a neural network model. The dataset contains various features of songs, such as danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and more. The goal is to classify songs into different popularity categories based on these features.

## Project Structure
- **Data Retrieval**: Attempted to retrieve song data using the Spotify API but faced rate-limiting issues. The dataset was then loaded from a CSV file.
- **Data Preprocessing**: Cleaned the dataset by removing duplicates and handling missing values. The 'popularity' column was binned into categories: 'flop', 'Low', 'Moderate', 'High', and 'Hit'.
- **Model Building**: Built and trained three different neural network models:
  1. **Model 1**: A simple neural network with no hidden layers.
  2. **Model 2**: A neural network with one hidden layer.
  3. **Model 3**: A neural network with two hidden layers.
- **Model Evaluation**: Compared the training and testing accuracies of the three models to determine the best-performing model.

## Dataset
The dataset used in this project contains the following features:
- `track_id`: Unique identifier for each track.
- `artists`: List of artists who performed the track.
- `album_name`: Name of the album the track belongs to.
- `track_name`: Name of the track.
- `popularity`: Popularity score of the track (0-100).
- `duration_ms`: Duration of the track in milliseconds.
- `explicit`: Whether the track contains explicit content.
- `danceability`: How suitable the track is for dancing.
- `energy`: Energy level of the track.
- `key`: Key the track is in.
- `loudness`: Overall loudness of the track in decibels (dB).
- `mode`: Modality of the track (major or minor).
- `speechiness`: Presence of spoken words in the track.
- `acousticness`: Acousticness of the track.
- `instrumentalness`: Instrumentalness of the track.
- `liveness`: Presence of an audience in the recording.
- `valence`: Positivity of the track.
- `tempo`: Tempo of the track in beats per minute (BPM).
- `time_signature`: Time signature of the track.
- `track_genre`: Genre of the track.

## Methodology
1. **Data Loading and Preprocessing**:
   - Loaded the dataset from a CSV file.
   - Removed duplicates and handled missing values.
   - Binned the 'popularity' column into categories.
   - Encoded categorical variables using `LabelEncoder`.
   - Split the dataset into training and testing sets.
   - Normalized the features using `StandardScaler`.

2. **Model Building**:
   - **Model 1**: A simple neural network with no hidden layers.
   - **Model 2**: A neural network with one hidden layer.
   - **Model 3**: A neural network with two hidden layers.
   - Compiled each model using the Adam optimizer and categorical cross-entropy loss.
   - Trained each model for 150 epochs.

3. **Model Evaluation**:
   - Compared the training and testing accuracies of the three models.
   - Visualized the training and validation accuracy and loss over epochs.

## Results
- **Model 1**: Achieved a validation accuracy of approximately 59.31%.
- **Model 2**: Achieved a validation accuracy of approximately 95.89%.
- **Model 3**: Achieved a validation accuracy of approximately 96.11%.

The results indicate that adding hidden layers to the neural network significantly improves the model's performance.

## Conclusion
Through this project, we learned:
- The importance of data preprocessing, including handling missing values, encoding categorical variables, and normalizing features.
- The impact of hidden layers on the performance of neural networks.
- The effectiveness of the Adam optimizer in training neural networks.
- The significance of evaluating model performance using validation accuracy and loss.

## Future Work
- Experiment with different neural network architectures and hyperparameters.
- Explore other machine learning algorithms, such as Support Vector Machines (SVM) or Random Forests, for comparison.
- Investigate feature importance to identify the most influential features in predicting song popularity.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `matplotlib`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spotify-song-popularity-prediction.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Jupyter notebook `Kodumuri_finalDL.ipynb` to execute the code and see the results.
2. Modify the notebook to experiment with different models and parameters.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Spotify for providing the dataset.
- TensorFlow and Keras for providing the tools to build and train neural networks.
- Scikit-learn for data preprocessing utilities.

---

This README provides a comprehensive overview of the project, including its structure, methodology, results, and future work. It also includes instructions for installation and usage, making it easy for others to replicate and build upon the work.