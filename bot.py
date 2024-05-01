import telebot
import uuid
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('mnist.h5')
bot = telebot.TeleBot('%BOT_TOKEN%')


def predict_digit(path):
    img = prepare_image(path)
    res = model.predict([img])[0]
    print(model.predict([img]))
    return np.argmax(res), max(res)


def prepare_image(path):
    image = Image.open(path)
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)

    image = image.reshape(1, 28, 28, 1)
    image = image/255.0
    return image


@bot.message_handler(content_types=['text', 'photo', 'image'])
def get_message(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id, 'Hi! This is the Numerizer - simple bot for digit recognition'
                                               ' from image. This bot uses simple convolutional neural network to'
                                               ' do it.\nEnter /help to see commands')
    elif message.text == '/help':
        bot.send_message(message.from_user.id, '''
            List of commands:
            /start - print information about this bot
            /help - print list of commands
            /numerize - try to recognize digit from image
        ''')
    elif message.text == '/numerize':
        bot.send_message(message.from_user.id, 'Great! Please, send image with digit which you want to recognize'
                                               '(I support only .jpg format files with compression)')
        bot.register_next_step_handler(message, get_image)
    else:
        bot.send_message(message.from_user.id, "I don't understand you :(\n Enter /help to see a list of commands")


def get_image(message):
    if not message.photo:
        bot.send_message(message.from_user.id, "This isn't a photo! Please, give me a photo, or enter /cancel, "
                                               "to return")
    else:
        file_id = message.photo[-1].file_id
        path = bot.get_file(file_id)
        downloaded_file = bot.download_file(path.file_path)

        extn = '.' + str(path.file_path).split('.')[-1]
        name = 'saved_images/' + str(uuid.uuid4()) + extn

        with open(name, 'wb') as new_file:
            new_file.write(downloaded_file)

        digit, accuracy = predict_digit(name)
        bot.send_message(message.from_user.id, f"I think, It's {digit} with {round(float(accuracy * 100), 2)}"
                                               f"% accuracy")


bot.polling(none_stop=True, interval=0)
