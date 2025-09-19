import sys
import importlib
from time import time
sys.path.append('/home/studente/venv/lib/python3.12/site-packages')

import broker
importlib.reload(broker)
import numpy as np
from tensorflow import keras
import tensorflow as tf

MAX_LENGTH = 8  # numero di bytes del payload che tengo

# load model from file
TLSmodel = keras.models.load_model('tf_models/autoencoderTLS_model.keras')

# set threshold
THRESHOLD_TLS = 0.20


def content_to_features(data, number_of_bytes=MAX_LENGTH):
    x = data.ljust(number_of_bytes * 2, "0")
    x = x[:(number_of_bytes * 2)]
    byte_values = []
    for i in range(0, len(x), 2):
        byte_values.append(chr(int(x[i:i + 2], 16)))

    bits = []
    for byte in byte_values:
        bits.append([(ord(byte) & (1 << i)) >> i for i in [7, 6, 5, 4, 3, 2, 1, 0]])

    return list(np.array(bits).astype(np.uint8).flat)


endpoint = broker.Endpoint()
subscription = endpoint.make_subscriber("tensorflow/content")
status_subscription = endpoint.make_status_subscriber(True)
endpoint.peer("192.168.1.76", 9999)

# Attendi lo stato PeerAdded
while True:
    status = status_subscription.get()  # Questo si blocca fino a quando un messaggio è disponibile
    if isinstance(status, broker.Status):
        print(f"Received status: {status.code()} - {status}")
        if status.code() == broker.SC.PeerAdded:
            print("Connected!")
            break
    else:
        print(f"Unexpected type for status: {type(status)}")
        sys.exit(1)

while True:
    (tag, data) = subscription.get()
    (src, sport, dst, dport, content) = broker.zeek.Event(data).args()

    # classify content
    reconstruction_TLS = TLSmodel.predict(np.array([content_to_features(content)]))
    loss_TLS = tf.keras.losses.mae(reconstruction_TLS, content_to_features(content))

    if loss_TLS > THRESHOLD_TLS:
        print(f"SRC: {src}:{sport} - DST: {dst}:{dport} - TYPE: Other")
    else:
        print(f"SRC: {src}:{sport} - DST: {dst}:{dport} - TYPE: TLS")
