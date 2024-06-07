import torch
import torchaudio
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

# Define class mappings
urban_sound_to_submission_mapping = {
     "asbfly",
     "ashdro1",
     "ashpri1",
     "ashwoo2",
     "asikoe2",
     "asiope1",
     "aspfly1",
     "aspswi1",
     "barfly1",
    
}

# Encode the class labels
class_labels = list(urban_sound_to_submission_mapping.keys())
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)

def predict(model, input, label_encoder):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0).item()
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label

# Load the sample submission file
sample_submission_path = '/mnt/data/sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)

if __name__ == "__main__":
    # Load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)

    # Load urban sound dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")

    # Iterate through the dataset and make predictions for the submission file
    for i in range(len(usd)):
        input, _ = usd[i][0], usd[i][1]  # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)

        predicted_label = predict(cnn, input, label_encoder)
        row_id = f"soundscape_{i*5}"

        # Update the corresponding row in the sample submission
        submission_column = urban_sound_to_submission_mapping[predicted_label]
        sample_submission.loc[sample_submission['row_id'] == row_id, submission_column] = 1.0

    # Save the updated sample submission file
    sample_submission.to_csv('updated_sample_submission.csv', index=False)
    print("Sample submission file updated and saved as 'updated_sample_submission.csv'")
