import librosa
import numpy as np
from scipy.linalg import sqrtm
import glob
from frechet_audio_distance import compute_fad

# all the wav file in the out folder
generated_audio_paths = glob.glob("../out/mus/*.wav")
reference_audio_paths = glob.glob("../data/audio/*.mp3")
# refernece audio path should be everything except the SIMPSON.mp3
reference_audio_paths = [path for path in reference_audio_paths if "SIMPSON" not in path]

# the two lists should have the same name, except one is wav and the other is mp3
# sort the list by the name of the audio file
generated_audio_paths.sort()
reference_audio_paths.sort()

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Extract 20 MFCCs
    return mfcc.mean(axis=1)  # Return the mean of each MFCC across time frames


def calculate_statistics(features):
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mean, cov


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check and correct imaginary numbers from computation
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sqrt(np.dot(diff, diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

# Example usage
ref_features = [extract_features(path) for path in reference_audio_paths]
gen_features = [extract_features(path) for path in generated_audio_paths]

ref_mean, ref_cov = calculate_statistics(ref_features)
gen_mean, gen_cov = calculate_statistics(gen_features)

distance = frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov)
# print the result to a txt file in the out folder caller FAD.txt
with open("../out/FAD.txt", "w") as f:
    f.write(f"Fréchet Audio Distance: {distance}")
    
# print("Fréchet Audio Distance:", distance)

