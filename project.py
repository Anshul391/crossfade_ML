import librosa
import numpy as np
import os
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import os


from sklearn.ensemble import GradientBoostingRegressor

def get_bpm(wav_file):
    sample_rate, audio_data = wavfile.read(wav_file)
        
    # If stereo, convert to mono by averaging channels
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio data
    audio_data = audio_data / (2.0**15) if audio_data.dtype == np.int16 else audio_data
    
    # Apply a low-pass filter to focus on the frequency range where beats typically occur
    nyquist = sample_rate / 2
    cutoff = 150 / nyquist  # 150 Hz
    b, a = signal.butter(4, cutoff, 'low')
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    # Calculate the envelope of the signal (amplitude)
    envelope = np.abs(filtered_audio)
    
    # Smooth the envelope
    win_size = int(sample_rate * 0.05)  # 50ms window
    if win_size % 2 == 0:  # Ensure odd window size for centered smoothing
        win_size += 1
    envelope = signal.savgol_filter(envelope, win_size, 2)

    # Downsample to reduce computation (one value per 100ms)
    hop_size = int(sample_rate * 0.05)  # 50ms hop
    envelope = envelope[::hop_size]
    effective_sr = sample_rate / hop_size
    
    # Calculate autocorrelation of the envelope
    # This helps find the periodicities in the signal
    corr = signal.correlate(envelope, envelope, mode='full')
    corr = corr[len(corr)//2:]  # Take only the positive lags
    
    # Calculate beat envelope using FFT
    # Find the dominant frequencies in our correlation
    fft = np.abs(np.fft.rfft(corr))
    freqs = np.fft.rfftfreq(len(corr), 1.0/effective_sr)
    
    # Limit to a reasonable BPM range (40-200 BPM)
    min_bpm, max_bpm = 40, 200
    min_idx = np.argmax(freqs > min_bpm/60)
    max_idx = np.argmax(freqs > max_bpm/60)
    if max_idx == 0:
        max_idx = len(freqs)
    
    # Find the peak in our target BPM range
    peak_idx = min_idx + np.argmax(fft[min_idx:max_idx])
    bpm = freqs[peak_idx] * 60
    
            
    return bpm

def apply_crossfade(song1, song2, sr, fade_duration=5.0):
    """
    Crossfade two audio tracks using a gradient boosting model for the fade curve.
    """
    # Compute fade samples
    fade_samples = int(fade_duration * sr)
    
    # Generate training data for the gradient model (simulating ideal fade curve)
    x = np.linspace(0, 1, fade_samples).reshape(-1, 1)
    y = np.sin(x * np.pi / 2)  # Example curve (smooth sine-based fade)
    
    # Train a gradient boosting model to learn this curve
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(x, y.ravel())
    
    # Generate fade curves
    fade_out = model.predict(x)
    fade_in = fade_out[::-1]  # Reverse fade-out curve for fade-in
    
    # Extract last fade_samples from song1 and first fade_samples from song2
    song1_end = song1[-fade_samples:]
    song2_start = song2[:fade_samples]
    
    # Apply fade effect
    song1_end = song1_end * fade_out[:, np.newaxis]
    song2_start = song2_start * fade_in[:, np.newaxis]
    
    # Merge tracks with crossfade
    crossfade = song1_end + song2_start
    output = np.concatenate([song1[:-fade_samples], crossfade, song2[fade_samples:]])
    
    return output

def train_model_on_folder(folder_path):
    """
    Train the model using multiple audio files in the given folder.
    """
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    audio_data = []
    
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        y, sr = librosa.load(file_path, sr=None, mono=True)
        audio_data.append((y, sr))
    
    if len(audio_data) < 2:
        raise ValueError("At least two audio files are required for training.")
    
    # Train and crossfade all consecutive pairs
    for i in range(len(audio_data) - 1):
        song1, sr1 = audio_data[i]
        song2, sr2 = audio_data[i + 1]
        
        if sr1 != sr2:
            raise ValueError("Sample rates of the audio files must match!")
        
        bpm1 = get_bpm(song1)
        bpm2 = get_bpm(song2)
        print(f"Processing {audio_files[i]} (BPM: {bpm1}) with {audio_files[i+1]} (BPM: {bpm2})")
        
        if abs(bpm1 - bpm2) > 5:
            print("Adjusting tempo...")
            song2 = librosa.effects.time_stretch(song2, bpm1 / bpm2)
        
        output = apply_crossfade(song1, song2, sr1)
        output_filename = f"crossfade_{i}.wav"
        sf.write(output_filename, output, sr1)
        print(f"Saved {output_filename}")

# Example usage
folder_path = "./edm_hse_id_001-004_wav"  # Change this to the actual folder path
train_model_on_folder(folder_path)
