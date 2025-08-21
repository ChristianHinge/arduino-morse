import tensorflow as tf
from pathlib import Path

# Representative dataset
def rep_ds_gen():
    import train  # reuse functions from train.py

    for x_batch, _ in train.train_ds.take(50):
        for x in x_batch:
            yield [tf.expand_dims(tf.cast(x, tf.float32), 0)]


converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_ds_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
Path("out").mkdir(exist_ok=True)
(Path("out") / "model_int8.tflite").write_bytes(tflite_model)
print("Exported out/model_int8.tflite")


# Export to C array
def tflite_to_cc(in_path="out/model_int8.tflite", out_path="out/model_data.cc", var="g_model"):
    data = Path(in_path).read_bytes()
    hex_bytes = ", ".join(f"0x{b:02x}" for b in data)
    cc = f"""#include <cstddef>
alignas(16) const unsigned char {var}[] = {{ {hex_bytes} }};
const int {var}_len = {len(data)};
"""
    Path(out_path).write_text(cc)
    print(f"Wrote {out_path} ({len(data)} bytes)")


tflite_to_cc()
