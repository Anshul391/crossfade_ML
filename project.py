import librosa
import soundfile as sf
import numpy as np
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_bpm(audio_data, sample_rate):
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data / np.max(np.abs(audio_data))
    nyquist = sample_rate / 2
    cutoff = 150 / nyquist
    b, a = signal.butter(4, cutoff, 'low')
    filtered_audio = signal.filtfilt(b, a, audio_data)
    envelope = np.abs(filtered_audio)
    win_size = int(sample_rate * 0.05)
    if win_size % 2 == 0:
        win_size += 1
    envelope = signal.savgol_filter(envelope, win_size, 2)
    hop_size = int(sample_rate * 0.05)
    envelope = envelope[::hop_size]
    effective_sr = sample_rate / hop_size
    corr = signal.correlate(envelope, envelope, mode='full')
    corr = corr[len(corr)//2:]
    fft = np.abs(np.fft.rfft(corr))
    freqs = np.fft.rfftfreq(len(corr), 1.0/effective_sr)
    min_bpm, max_bpm = 40, 200
    min_idx = np.argmax(freqs > min_bpm/60)
    max_idx = np.argmax(freqs > max_bpm/60)
    if max_idx == 0:
        max_idx = len(freqs)
    peak_idx = min_idx + np.argmax(fft[min_idx:max_idx])
    bpm = freqs[peak_idx] * 60
    return bpm

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

def train_fade_duration_model(bpm_values, fade_durations):
    X = np.array(bpm_values).reshape(-1, 1)
    y = np.array(fade_durations)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model

def filter_trailing_zeros(arr):
    """
    Filters out trailing zero values from a NumPy array.

    Args:
        arr (np.ndarray): The input NumPy array.

    Returns:
        np.ndarray: A new array with trailing zeros removed.
    """
    non_zero_indices = np.nonzero(arr)[0]
    if non_zero_indices.size > 0:
      last_non_zero_index = non_zero_indices[-1]
      return arr[:last_non_zero_index + 1]
    else:
      return np.array([])

def apply_crossfade(song1, song2, sr, fade_duration):
    fade_samples = int(fade_duration * sr)
    song1_end = song1[-fade_samples:]
    song2_start = song2[:fade_samples]
    fade_out = np.linspace(1, 0, fade_samples)
    fade_in = np.linspace(0, 1, fade_samples)
    if len(song1_end.shape) > 1:
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]
    song1_end = song1_end * fade_out
    song2_start = song2_start * fade_in
    crossfade = song1_end + song2_start
    output = np.concatenate([song1[:-fade_samples], crossfade, song2[fade_samples:]])
    return output

def train_model_on_folder(folder_path, output_folder="./crossfades"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    audio_data = []
    fade_durations = []
    bpm_values = []
    for file in audio_files[::80]:
        file_path = os.path.join(folder_path, file)
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            bpm = get_bpm(y, sr)
            fade_duration = np.clip(5 - (bpm / 40), 1, 5)
            audio_data.append((y, sr, file))
            bpm_values.append(bpm)
            fade_durations.append(fade_duration)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    model = train_fade_duration_model(bpm_values, fade_durations)
    return model, bpm_values, audio_data

def predict_fade_duration_for_songs(model, song1_path, song2_path):
    # Load songs
    song1, sr1 = librosa.load(song1_path, sr=None, mono=True)
    song2, sr2 = librosa.load(song2_path, sr=None, mono=True)

    song1 = filter_trailing_zeros(song1)

    # Predict BPM for both songs
    bpm1 = get_bpm(song1, sr1)
    bpm2 = get_bpm(song2, sr2)

    # Predict fade durations using the model
    fade_duration1 = model.predict(np.array([[bpm1]]))[0]
    fade_duration2 = model.predict(np.array([[bpm2]]))[0]

    print(f"Predicted fade duration for song 1 (BPM {bpm1}): {fade_duration1} seconds")
    print(f"Predicted fade duration for song 2 (BPM {bpm2}): {fade_duration2} seconds")

    return fade_duration1, fade_duration2

if __name__ == "__main__":
    folder_path = "./edm_hse_id_001-004_wav"
    model, bpm_values, audio_data = train_model_on_folder(folder_path)

    # Paths to your songs
    song1_path = "Igottafeeling.wav"
    song2_path = "moveslikejagger.wav"

    # Predict fade durations for these two songs
    fade_duration1, fade_duration2 = predict_fade_duration_for_songs(model, song1_path, song2_path)

    # Apply the crossfade with predicted fade durations
    song1, sr1 = librosa.load(song1_path, sr=None, mono=True)
    song2, sr2 = librosa.load(song2_path, sr=None, mono=True)

    output = apply_crossfade(song1, song2, sr1, fade_duration1)
    sf.write("Output.wav", output, sr1)
    print(f"Fade duration for crossfade applied: {fade_duration1}")
