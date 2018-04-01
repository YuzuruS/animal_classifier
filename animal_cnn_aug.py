import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def main():
	X_train, X_test, Y_train, Y_test = np.load("./animap_aug.npy")
	# 正規化(0-1の間のほうがブレが少ない)
	X_train = X_train.astype("float") / 256
	X_test = X_test.astype("float") / 256

	# 正解地は1、他は0 one-hot-vector
	Y_train = np_utils.to_categorical(Y_train, num_classes)
	Y_test = np_utils.to_categorical(Y_test, num_classes)

	model = model_train(X_train, Y_train)
	model_eval(model, X_test, Y_test)

def model_eval(model, X, Y):
	scores = model.evaluate(X, Y, verbose=1)
	print('Test Loss: ', scores[0])
	print('Test Accuracy: ', scores[1])

def model_train(X, Y):
	model = Sequential()
	# ニューラルネットワークの層を足す
	# 畳み込みの結果が同じになるように
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))
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

	# バッチサイズ=1回のトレーニングに使うデータの数 32枚
	# トレーニングを何セットやるのか 100回
	model.fit(X,Y,batch_size=32, epochs=100)

	# モデルの保存
	model.save('./animal_cnn_aug.h5')

	return model

if __name__ == "__main__":
	main()