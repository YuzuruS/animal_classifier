from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras, sys
import numpy as np
from PIL import Image

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def build_model():
	model = Sequential()
	# ニューラルネットワークの層を足す
	# 畳み込みの結果が同じになるように
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3)))
	# 活性化関数
	model.add(Activation('relu'))
	# 二層目
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	# 三層目
	# 一番大きい値を取り出す。特徴を際立たせる
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 25％以下は捨てる データの偏りは捨てる
	model.add(Dropout(0.25))

	# 64のフィルターを持っている
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 25％以下は捨てる データの偏りは捨てる
	model.add(Dropout(0.25))

	# データを一列に並べる
	model.add(Flatten())
	#全結合層
	model.add(Dense(512))

	# 半分捨てる
	model.add(Dropout(0.5))

	#最後の出力層のノード数
	model.add(Dense(3))
	model.add(Activation('softmax'))

	# トレーニング時の更新アルゴリズム
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# 損失関数。正解と推定値との誤差
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# モデルのロード
	model = load_model('./animal_cnn_aug.h5')

	return model

def main():
	image = Image.open(sys.argv[1])
	image = image.convert('RGB')
	image = image.resize((image_size, image_size))
	data = np.asarray(image)
	X = []
	X.append(data)
	X = np.array(X)
	model = build_model()

	result = model.predict([X])[0]
	predicted = result.argmax()
	percentage = int(result[predicted] * 100)
	print("{0}({1}%)".format(classes[predicted], percentage))

if __name__ == "__main__":
	main()