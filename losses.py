import tensorflow as tf
def softmax_focal_loss(label, pred, class_num=30, gamma=2):

    label = tf.squeeze(tf.cast(tf.one_hot(tf.cast(label, tf.int32), class_num), pred.dtype))
    pred = tf.clip_by_value(pred, 1e-8, 1.0)
    w1 = tf.math.pow((1.0 - pred), gamma)
    L = - tf.math.reduce_sum(w1 * label * tf.math.log(pred))
    return L

def focal_loss(pred, y, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    zeros = tf.zeros_like(pred, dtype=pred.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = tf.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)
import os
dir = '/data2/competition/classification/train_extern'
for dirpath, dirnames, filenames in os.walk(dir):
    for dirname in dirnames:
        for path, _, imgnames in os.walk(os.path.join(dirpath, dirname)):
            for imgname in imgnames:
                with open(os.path.join(path, imgname), 'rb') as imageFile:
                    if imageFile.read().startswith(b'RIFF'):
                        print(f"{os.path.join(path, imgname)} - found!")
                    # first4 = tuple(imageFile.read(4))
                    # if first4 == (0x47, 0x49, 0x46, 0x38):
                    #     print(f"{os.path.join(path, imgname)} - found!")
