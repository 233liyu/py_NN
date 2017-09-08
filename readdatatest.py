# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

#f1 = open("MNIST_data/train-labels.csv", "r")

dataset_path = "MNIST_data/"
test_labels_file = "test-labels.csv"
train_labels_file = "train-labels.csv"

test_set_size = 5

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 3
BATCH_SIZE = 5


def encode_label(label):
    return int(label)


def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(",")
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels


# reading labels and file path
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

# transform relative path into full path
train_filepaths = [dataset_path + fp for fp in train_filepaths]
test_filepaths = [dataset_path + fp for fp in test_filepaths]

# for this example we will create or own test partition
all_filepaths = train_filepaths + test_filepaths
all_labels = train_labels + test_labels

all_filepaths = all_filepaths[:20]
all_labels = all_labels[:20]

# convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# create a partition vector
partitions = [0] * len(all_filepaths)
partitions[:test_set_size] = [1] * test_set_size
random.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

# create input queues
train_input_queue = tf.train.slice_input_producer(
    [train_images, train_labels],
    shuffle=False)
test_input_queue = tf.train.slice_input_producer(
    [test_images, test_labels],
    shuffle=False)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
    [train_image, train_label],
    batch_size=BATCH_SIZE
    # ,num_threads=1
)
test_image_batch, test_label_batch = tf.train.batch(
    [test_image, test_label],
    batch_size=BATCH_SIZE
    # ,num_threads=1
)

print ("input pipeline ready")

with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.initialize_all_variables())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print
    "from the train set:"
    for i in range(20):
        print
        sess.run(train_label_batch)

    print
    "from the test set:"
    for i in range(10):
        print
        sess.run(test_label_batch)

        # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()