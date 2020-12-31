import keras.backend as K
import tensorflow as tf

epsilon = tf.keras.backend.epsilon()


def f1_score(y_true, y_pred):  # taken from old keras source code
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	recall = true_positives / (possible_positives + K.epsilon())
	f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
	return f1_val


def make_IoU(threshold=0.5):
	def IoU(y_true, y_pred):
		p = tf.dtypes.cast(y_pred > threshold, tf.float32)
		Intersection = tf.dtypes.cast(tf.math.equal((y_true * p), 1.0), tf.float32)
		union = tf.dtypes.cast(tf.math.greater_equal((y_true + p), 1.0), tf.float32)

		Intersection = tf.reduce_sum(Intersection)
		union = tf.reduce_sum(union)

		if union.numpy() == 0:
			iou = 1.0
		else:
			iou = Intersection / union
		return iou

	return IoU


def make_categorical_IoU(label, threshold=0.5):
	IoUs = []
	for i in range(label.n_labels):
		def IoU(y_true, y_pred, i=i):
			y_true = y_true[:, :, :, i]
			y_pred = y_pred[:, :, :, i]
			p = tf.dtypes.cast(y_pred > threshold, tf.float32)
			Intersection = tf.dtypes.cast(tf.math.equal((y_true * p), 1.0), tf.float32)
			union = tf.dtypes.cast(tf.math.greater_equal((y_true + p), 1.0), tf.float32)

			Intersection = tf.reduce_sum(Intersection)
			union = tf.reduce_sum(union)

			if union.numpy() == 0:
				iou = 1.0
			else:
				iou = Intersection / union
			return iou

		IoU.__name__ = "IoU_" + label.name[i]
		IoUs.append(IoU)
	return IoUs


def make_F1score(threshold=0.5):
	def F1score(y_true, y_pred):
		p = tf.dtypes.cast(y_pred > threshold, tf.float32)

		A = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
		A = tf.reduce_sum(A)
		B = tf.reduce_sum(p)

		Intersection = tf.dtypes.cast(tf.math.equal((y_true * p), 1.0), tf.float32)
		Intersection = tf.reduce_sum(Intersection)

		denominator = (A + B).numpy()
		if denominator == 0:
			F1 = 1.0
		else:
			F1 = 2 * Intersection / denominator
		return F1

	return F1score


def make_categorical_F1score(label, threshold=0.5):
	F1scores = []
	for i in range(label.n_labels):
		def F1score(y_true, y_pred, i=i):
			y_true = y_true[:, :, :, i]
			y_pred = y_pred[:, :, :, i]

			p = tf.dtypes.cast(y_pred > threshold, tf.float32)
			A = tf.dtypes.cast(tf.math.equal(y_true, 1.0), tf.float32)
			A = tf.reduce_sum(A)
			B = tf.reduce_sum(p)

			Intersection = tf.dtypes.cast(tf.math.equal((y_true * p), 1.0), tf.float32)
			Intersection = tf.reduce_sum(Intersection)

			denominator = (A + B).numpy()
			if denominator == 0:
				F1 = 1.0
			else:
				F1 = 2 * Intersection / denominator
			return F1

		F1score.__name__ = "F1score_" + label.name[i]
		F1scores.append(F1score)
	return F1scores
