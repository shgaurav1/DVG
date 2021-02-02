from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


import tensorflow as tf
import pdb
import numpy as np
from frechet_video_distance import *

# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 16
VIDEO_LENGTH = 15





real_video = torch.load('ucf_test_x_0.pt')
generated_video = torch.load('4gen_seq_ours_lstm_ucf 0.pt')#4gen_seq_svg_ucf 4gen_seq_ours_lstm_ucf
# pdb.set_trace()
real_video = torch.stack(real_video)
real_video = real_video[20:35]


new_seq = [torch.Tensor(seq) for seq in generated_video]
# seq = generated_video.transpose(0,1)#
seq = torch.stack(new_seq)

# seq = seq.transpose(0,1).transpose(4,3).transpose(3,2)#
seq = (seq[15:30] + 1)/2
seq = seq.transpose(0,1).transpose(2,3).transpose(3,4)
real_video = real_video.transpose(0,1).transpose(2,3).transpose(3,4)
# real_video = torch.cat([real_video,real_video,real_video],4)
# seq = torch.cat([seq,seq,seq],4)
seq = seq*255
real_video = real_video*255
# pdb.set_trace()


# print(real_video.shape,seq.shape)
# x_tf = tf.convert_to_tensor(real_video.cpu().numpy(),np.float32)
# fake = tf.convert_to_tensor(seq.cpu().numpy(),np.float32)

# calculate_fvd(create_id3_embedding(preprocess(x_tf,(64,64))),create_id3_embedding(preprocess(fake,(64,64))))



def main(argv):
  del argv
  with tf.Graph().as_default():

    # first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3])
    # second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]
    #                               ) * 255
    x_tf = tf.convert_to_tensor(real_video[0:16].cpu().numpy(),np.int32)
    fake = tf.convert_to_tensor(seq[0:16].cpu().numpy(),np.int32)
    result = calculate_fvd(
        create_id3_embedding(preprocess(x_tf,
                                                (224, 224))),
        create_id3_embedding(preprocess(fake,
                                                (224, 224))))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("FVD is: %.2f." % (sess.run(result)/16))


if __name__ == "__main__":
  tf.app.run(main)
