import tensorflow as tfimport keras.backend as kbdef mse(y_true, y_pred):    """    Computes the MSE error between a set of true parameters (as generated from a prior)    and a set of predicted values (as generated by the nn approximator).    :param y_true: tf.Tensor of shape (batch_size, n_out_dim) -- the vector of predicted values    :param y_pred: tf.Tensor of shape (,) -- a single scalar value representing thr heteroskedastic loss    :return: tf.Tensor of shape (,) -- average loss over batch_size    """    square_differences = kb.square(y_true - y_pred)    mse = kb.mean(square_differences, axis=-1)    return msedef heteroskedastic_loss(y_true, y_pred):    """    Computes the heteroskedastic loss between a set of true parameters (as generated from a prior)    and a set of predicted values (as generated by the nn approximator).    :param y_pred: tf.Tensor of shape (batch_size, n_out_dim) -- the vector of predicted values    :param y_true: tf.Tensor of shape (batch_size, n_out_dim) -- the vector of true values    :return: tf.Tensor of shape (,) -- a single scalar value representing thr heteroskedastic loss    """    y_mean, y_var = tf.split(y_pred, 2, axis=-1)    logvar = tf.reduce_sum(input_tensor=0.5 * y_var, axis=-1)    squared_error = tf.reduce_sum(input_tensor=0.5 * tf.square(y_true - y_mean) / tf.exp(y_var), axis=-1)    loss = tf.reduce_mean(input_tensor=squared_error + logvar)    return loss