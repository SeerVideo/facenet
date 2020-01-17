import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from glob import glob
from tqdm import tqdm


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = args.batch_size

    with tf.Graph().as_default():

        with tf.Session(config=config) as sess:

            facenet.load_model(args.model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            for dir_base in tqdm(sorted(glob(args.input_dir+'/n*'))):

                image_files = sorted(glob('{}/*.png'.format(dir_base)))

                emb_array = np.zeros((len(image_files), embedding_size))
                for i in range(0, len(image_files), batch_size):
                    paths_batch = image_files[i:i+batch_size]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[i:i+batch_size,:] = sess.run(embeddings, feed_dict=feed_dict)

                fn = dir_base.split('/')[-1]
                np.save(os.path.join(args.output_dir, fn), emb_array)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/home/seercv/facenet_assets/vggface2_a',
                        help='Directory with aligned images.')

    parser.add_argument('--output_dir', type=str,
                        default='/home/seercv/facenet_assets/data',
                        help='Directory to store npy arrays of embeddings.')

    parser.add_argument('--model_path', type=str,
                        default='/home/seercv/facenet_assets/models/20180402-114759',
                        help='Path to pretrained model.')

    parser.add_argument('--image_size', type=int, default=160,
                        help='Image size of aligned image.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of samples to put on gpu.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
