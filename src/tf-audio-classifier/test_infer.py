import numpy as np
import tensorflow as tf
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter

# -------------------------
# Load model
# -------------------------
interpreter = Interpreter(model_path="out/model_int8.tflite")
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]
in_scale, in_zero = in_details["quantization"]
out_scale, out_zero = out_details["quantization"]

classes = Path("labels.txt").read_text().splitlines()

# -------------------------
# Preprocess single wav
# -------------------------
import train  # reuse preprocessing
def preprocess_wav_file(path):
    wav = train.load_audio(tf.constant(str(path)))
    feats = train.wav_to_features(wav)
    return feats.numpy()

# -------------------------
# Run inference
# -------------------------
def run_file(path):
    feats = preprocess_wav_file(path)
    q = np.round(feats / in_scale + in_zero).astype(np.int8)
    q = np.expand_dims(q, 0)
    interpreter.set_tensor(in_details["index"], q)
    interpreter.invoke()
    out_q = interpreter.get_tensor(out_details["index"])[0]
    probs = (out_q.astype(np.float32) - out_zero) * out_scale
    idx = int(np.argmax(probs))
    return classes[idx], probs


pred, probs = run_file("data/yes/example.wav")  # replace with your test file
print("Prediction:", pred, "Probs:", probs)
