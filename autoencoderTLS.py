import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense
import numpy as np



MAX_LENGTH = 8 #numero di bytes del payload che tengo, dopo aver provato vari valori 8 è quello che fornisce la miglior performance
def content_to_features(data, number_of_bytes = MAX_LENGTH):
    x = data.ljust(number_of_bytes * 2, "0")
    x = x[:(number_of_bytes * 2)]
    
    byte_values = []
    for i in range(0, len(x), 2):
        byte_values.append(chr(int(x[i:i+2], 16)))
    bits = []
    for byte in byte_values:
        bits.append([(ord(byte) & (1<<i)) >>i for i in [7, 6, 5, 4, 3, 2, 1, 0]])
    
    return list(np.array(bits).astype(np.uint8).flat)



with open('./training_data/TLS', 'r') as f:
    raw_data = [content_to_features(line) for line in f.readlines()]

x_data = np.array(raw_data).astype(np.uint8)
#print(x_data.shape)



x_data = tf.random.shuffle(x_data, seed=100)

VALIDATION_SET_SIZE = round(0.1 * x_data.shape[0])  # 10%
TEST_SET_SIZE = round(0.2 * x_data.shape[0])  # 20%
TRAIN_SET_SIZE = x_data.shape[0] - VALIDATION_SET_SIZE - TEST_SET_SIZE  # 70%

x_train, x_val, x_test = tf.split(x_data, [TRAIN_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE])



input_dimensions = MAX_LENGTH * 8   #numero di input = numero di bit

model = Sequential()
model.add(Dense(input_dimensions, input_shape=(input_dimensions,)))
model.add(Dense(input_dimensions // 2, activation='relu'))
model.add(Dense(input_dimensions // 4, activation='relu'))
model.add(Dense(input_dimensions // 4, activation='relu'))
model.add(Dense(input_dimensions // 2, activation='relu'))
model.add(Dense(input_dimensions, activation='relu'))
# funzione di attivazione relu plausibile in quando ho dei valori binari



model.compile(optimizer='adam', loss='mae')
model.fit(x_train, x_train, batch_size=128, epochs=10)  #labels sono sempre le features stesse nell'autoencoder



import matplotlib.pyplot as plt
plt.plot(tf.losses.mae(x_val, model.predict(x_val)))
plt.show()



with open('./training_data/HTTP', 'r') as f:
    raw_data = [content_to_features(line) for line in f.readlines()]



http_data = np.array(raw_data).astype(np.uint8)
#print(http_data.shape)



""" porta 22: SSH (Secure Shell)
porta 53: DNS
porta 1119: servizi di gaming online """
ports = [22, 53, 1119]

raw_data = []

for port in ports:
    file_path = './training_data/' + str(port)
    with open(file_path, 'r') as f:
        raw_data.extend([content_to_features(line) for line in f.readlines()])

other_data = np.array(raw_data).astype(np.uint8)
#print(other_data.shape)



not_tls_data = np.concatenate((http_data, other_data), axis=0)



#fase tuning iperparametro attraverso validation set
plt.plot(tf.losses.mae(x_val, model.predict(x_val)))
plt.plot(tf.losses.mae(not_tls_data, model.predict(not_tls_data)))
plt.show()



#soglia individuata
THRESHOLD = 0.20



#fase test
loss_x_test = tf.losses.mae(x_test, model.predict(x_test))
loss_not_tls = tf.losses.mae(not_tls_data, model.predict(not_tls_data))

plt.plot(loss_x_test)
plt.plot(loss_not_tls)
plt.show()



FN = np.sum(loss_x_test > THRESHOLD)
TN = np.sum(loss_not_tls > THRESHOLD)
TP = len(loss_x_test) - FN
FP = len(loss_not_tls) - TN

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * TP / (2 * TP + FP + FN)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)



#riaddestra con tutti i dati (compreso validation e test set) prima di esportare
model.fit(x_data, x_data, batch_size=128, epochs=10)
model.save('autoencoderTLS_model.keras')