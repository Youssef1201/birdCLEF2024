
class BirdClefDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # Création d'un label encoder pour rendre mes labels (nom d'oiseau) en tenseur
        all_labels = []
        for dirname, _, filenames in os.walk(audio_dir):
            dirname = dirname.replace('/kaggle/input/birdclef-2024/train_audio','')
            all_labels.append(dirname.replace('/',''))
        all_labels.pop(0)
        self.label = LabelEncoder()
        self.label.fit(all_labels)
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        
        signal = signal.to(self.device)
        label_encoded = self.label.transform([label])
        label_tensor = torch.tensor(label_encoded).to(self.device)

        return signal, label_tensor

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        filename = self.annotations.iloc[index, -1]
        path = os.path.join(self.audio_dir, filename)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 0]

# Initialisation des paramètres
ANNOTATIONS_FILE = "/kaggle/input/birdclef-2024/train_metadata.csv"
AUDIO_DIR = "/kaggle/input/birdclef-2024/train_audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
NUM_CLASSES = len(pd.read_csv(ANNOTATIONS_FILE)['primary_label'].unique())

# Configuration du dispositif
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Transformation en Mel Spectrogram
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

# Chargement des données
dataset = BirdClefDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Afficher la taille du dataset
print(f"There are {len(dataset)} samples in the dataset.")
signal, label = dataset[6000]
print("matrice de l'oiseau: ", signal.shape, " c'est un: ", label)
