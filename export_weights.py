import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("fires_model.keras")
weights = model.get_weights()

np.savez(
    "mlp_weights.npz",
    W1=weights[0], b1=weights[1],
    W2=weights[2], b2=weights[3],
    W3=weights[4], b3=weights[5],
    W4=weights[6], b4=weights[7]
)

print("mlp_weights.npz 저장 완료")