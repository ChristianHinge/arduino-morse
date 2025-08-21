import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np

# -------------------------
# Config
# -------------------------
DATA_DIR = Path("data")
SAMPLE_RATE = 16000
DURATION_S = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_S)

FRAME_LEN = 640      # 40ms
FRAME_STEP = 320     # 20ms
NUM_MELS = 40
USE_MFCC = True
NUM_MFCC = 10

BATCH_SIZE = 64
EPOCHS = 20

# -------------------------
# Dataset setup
# -------------------------
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
class_to_id = {c: i for i, c in enumerate(classes)}
print("Classes:", classes)


def list_files():
    files, labels = [], []
    for c in classes:
        for p in (DATA_DIR / c).glob("*.wav"):
            files.append(str(p))
            labels.append(class_to_id[c])
    return tf.constant(files), tf.constant(labels, dtype=tf.int32)


files, labels = list_files()


def load_audio(path):
    audio = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)

    wav = tf.cond(
        tf.shape(wav)[0] < NUM_SAMPLES,
        lambda: tf.pad(wav, [[0, NUM_SAMPLES - tf.shape(wav)[0]]]),
        lambda: wav[:NUM_SAMPLES],
    )
    return wav


def wav_to_features(wav):
    stft = tf.signal.stft(
        wav, frame_length=FRAME_LEN, frame_step=FRAME_STEP, fft_length=FRAME_LEN
    )
    mag = tf.abs(stft)
    mel_fb = tf.signal.linear_to_mel_weight_matrix(
        NUM_MELS, mag.shape[-1], SAMPLE_RATE, 20.0, 4000.0
    )
    mel = tf.tensordot(mag, mel_fb, 1)
    mel = tf.math.maximum(mel, 1e-6)
    logmel = tf.math.log(mel)

    if USE_MFCC:
        feats = tf.signal.mfccs_from_log_mel_spectrograms(logmel)[..., :NUM_MFCC]
    else:
        feats = logmel

    mean = tf.reduce_mean(feats)
    std = tf.math.reduce_std(feats) + 1e-6
    feats = (feats - mean) / std
    feats = tf.expand_dims(feats, -1)
    return feats


def load_and_preprocess(path, label):
    wav = load_audio(path)
    x = wav_to_features(wav)
    return x, tf.one_hot(label, depth=len(classes))


ds = tf.data.Dataset.from_tensor_slices((files, labels))
ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

val_size = int(len(files) * 0.1)
train_ds = ds.skip(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = ds.take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------
# Model
# -------------------------
for x_batch, _ in train_ds.take(1):
    input_shape = x_batch.shape[1:]


def tiny_kws(input_shape, num_classes):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.Conv2D(24, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.Conv2D(32, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)


model = tiny_kws(input_shape, len(classes))
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# Save model
model.save("saved_model", include_optimizer=False)
(Path("labels.txt")).write_text("\n".join(classes))
print("Saved trained model and labels.")
