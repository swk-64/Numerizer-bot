import tensorflow as tf
from PIL import Image


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
for i in range(50):
    l_img = x_test[i]
    img1 = Image.fromarray(l_img, mode='L')
    img1 = img1.resize((300, 300))
    img1.save(f'image-{i + 1}-{y_test[i]}.jpg', 'JPEG')
