
def jaccard_distance(smooth=50):

    def jaccard_distance_fixed(y_true, y_pred):
        """
        Calculates mean of Jaccard distance as a loss function
        """
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jd =  (1 - jac) * smooth
        return tf.reduce_mean(jd)

    return jaccard_distance_fixed
