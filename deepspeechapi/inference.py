from deepspeech import Model
import scipy.io.wavfile as wav

model_path = 'assets/200output_graph.pb'
alphabet_path = 'assets/alpha_small.txt'
ds = Model(model_path, 26, 9, alphabet_path, 500)

# lm_path = '../lm.binary'

aud = 'media/test.wav'
def infer(audio_path):
    try:
        print(audio_path)
        fs, audio = wav.read(audio_path)
        processed_data = ds.stt(audio, fs)
        return processed_data
    except ValueError as e:
        return e


print(infer(aud))
