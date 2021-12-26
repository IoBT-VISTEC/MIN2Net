import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


def mean_squared_error(y_true, y_pred):
	""" loss function computing MSE of non-blank(!=0) in y_true
		Args:
			y_true(tftensor): true label
			y_pred(tftensor): predicted label
		return:
			MSE reconstruction error for loss computing
	"""
	loss = K.switch(K.equal(y_true, tf.constant(0.)),tf.zeros(K.shape(y_true)),K.square(y_pred - y_true))
	return K.mean(loss, axis=-1)

def triplet_loss(margin = 1.0):
    def inner_triplet_loss_objective(y_true, y_pred):
        labels = y_true
        embeddings = y_pred
        return tfa.losses.triplet_semihard_loss(y_true=labels, y_pred=embeddings,margin=margin)
    return inner_triplet_loss_objective

def SparseCategoricalCrossentropy(class_weight = None):
    """[SparseCategoricalCrossentropy]

    Args:
        class_weight ([dict], optional): dict of class_weight
        class_weight = {0: 0.3,
                        1: 0.7}
        Defaults to None.
    """
    def inner_sparse_categorical_crossentropy(y_true, y_pred):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        if class_weight:
            keys_tensor = tf.cast(tf.constant(list(class_weight.keys())), dtype=tf.int32)
            vals_tensor = tf.constant(list(class_weight.values()), tf.float32)
            input_tensor = tf.cast(y_true, dtype=tf.int32)
            init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
            table = tf.lookup.StaticHashTable(init, default_value=-1)
            sample_weight = table.lookup(input_tensor)
        else:
            sample_weight = None
        return scce(y_true, y_pred, sample_weight)
    return inner_sparse_categorical_crossentropy