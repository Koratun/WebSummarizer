# This will be the part of the program that takes data from the webscraper and summarizes it.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy


words = []

# Read the dictionary file and split on newlines into a list
with open("WebScraper/words.txt", "r") as wordfile:
    words = wordfile.read().splitlines()

# Add individual punctuation marks to the dictionary for it to recognize
punctuation = [p for p in "`~!@#$%^&*()_+-=[]{}“”|;':\",./\\ <>?—"]

for p in punctuation:
    words.insert(0, p)



# Load the generator and predict a summary for the given article
def summarize(article):
    gen = keras.models.load_model("./Generator")
    return decode_generator(gen.predict(article))


# Call with the output of the generator to decode it
def decode_generator(encoded_summary):
    summary = ''
    # Turns each entry in the encoded summary into its proper word from the dictionary. (and adds a space)
    for i in encoded_summary:
        summary += words[i] + ' '

    return summary



# Create a tensor to contain the dictionary
words_tensor = tf.convert_to_tensor(words, dtype=tf.string, name="Dictionary")


@tf.function
@keras.utils.register_keras_serializable(package="WebScraper")
def standardize_with_dictionary(data):
    # Will split data (one string) into a list containing many strings (that have been standardized) that need to be recombined into one string.
    unstandardized_example = split_against_dictionary(data, from_standardize=True)

    # Combine all strings back into a single string
    return tf.strings.reduce_join(unstandardized_example)



@tf.function
@keras.utils.register_keras_serializable(package="WebScraper")
# function for TV layer to call when it is splitting text it sees
def split_against_dictionary(data, from_standardize = False):
    out = []

    print(tf.unstack(tf.slice(tf.shape(data), [0], [1]))[0])
    # If this is called from the standardize function, we must be getting a single string string.
    # Convert it to a list containing a single entry to match what the split function will receive normally.
    if from_standardize:
        data = tf.reshape(data, [1, 1])

    # Get a single character from a string tensor
    def get_char_from_stringtensor(x, pos):
        return tf.strings.substr(x, pos, 1)

    def insertval_into_tensorarray(ta, pos, val):
        newta = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False)
        for i in range(pos):
            newta.write(i, ta.read(i))
        newta.write(newta.size(), val)
        for i in range(pos+1, ta.size()):
            newta.write(i, ta.read(i))
        return newta

    # Split by spaces, but keep the spaces as individual characters in the list.
    x = tf.Variable(0)
    print(data)
    for article in data:
        print(article)
        out.append([])
        aggregate = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False)
        print(tf.unstack(tf.strings.length(article))[0])
        for i in tf.range(tf.unstack(tf.strings.length(article))[0]):
            s = get_char_from_stringtensor(article, i)
            if s == ' ' or (aggregate.size() == 1 and aggregate.read(0) == ' '):
                out[x.eval()].append(tf.strings.reduce_join(aggregate.concat()))
                aggregate = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False)
            aggregate.write(aggregate.size(), s)
        # Append the last bit of trailing data in the aggregate.
        out[x].append(tf.strings.reduce_join(aggregate.concat()))
        x.assign_add(1)
    # This should be a list of lists containing strings.
    
    # A couple of short tensor functions to make the code easier to understand
    # Returns a tensor containing 1 if a match was found, 0 if not
    def find_stringtensor_in_tensorlist(x):
        return tf.slice(tf.shape_n(tf.where(tf.strings.regex_full_match(words_tensor, x))), [0], [1])


    def tensor_check_lower(x):
        # Check if the word is found when made lowercase, if so, return the lowercase tensor, otherwise continue down the rabbit hole.
        return tf.cond(tf.less(tf.constant([0]), find_stringtensor_in_tensorlist(tf.strings.lower(x))), tf.strings.lower, separate_punctuation)


    def separate_punctuation(x):
        substring_tensor = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False)
        aggregate = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False)
        for i in range(tf.strings.length(x)):
            s = get_char_from_stringtensor(x, i)
            for p in punctuation:
                # separate x by each match of a punctuation mark.
                if s == p or (aggregate.size() == 1 and aggregate.read(0) == p):
                    substring_tensor.append(tf.strings.reduce_join(aggregate.concat()))
                    aggregate = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False)
                    break
            aggregate.write(aggregate.size(), s)
        # Append the last bit of trailing data in the aggregate.
        substring_tensor.append(tf.strings.reduce_join(aggregate.concat()))
        # Concat substr tensor array into single tensor
        return substring_tensor.concat()


    for article in out:
        # article is list of single-string tensors
        for i in range(len(article)-1, -1, -1):
            # s is string tensor
            s: tf.Tensor = article[i]
            # Check if s is a space. Avoid many calculations this way.
            if s == ' ':
                continue
            # Check if word matches dictionary with no changes.
            # If word is found, return the tensor as is, otherwise continue down the rabbit hole.
            # output tensor will be a tensor with the same contents as s, a tensor of the same size but different contents, or a tensor with an extra dimension.
            output_tensor = tf.cond(tf.less(tf.constant([0]), find_stringtensor_in_tensorlist(s)), tf.identity, tensor_check_lower)

            if s.shape == output_tensor.shape:
                # Replace tensor regardless of contents
                article.pop(i)
                article.insert(i, output_tensor)
            else:
                # If output tensor is larger than the original tensor, then we need to do add each element of the output tensor independently to article.
                article.pop(i)
                for j in range(len(tf.shape_n(output_tensor))-1, -1, -1):
                    article.insert(i, tf.slice(output_tensor, [j], [1]))


    # out is now ready to be returned to the function that called it.
    if from_standardize:
        # If called by standardize, out should be a list with one list in it.
        return tf.convert_to_tensor(out[0])
    else:
        # If called by the vectorization, then it should return all the data as is.
        return tf.ragged.stack([tf.unstack(t) for t in out], axis=0).merge_dims(0, 1)
            

# Create a text vectorization layer for both the generator and the discriminator to use
tv = layers.experimental.preprocessing.TextVectorization(vocabulary=words)#, standardize=standardize_with_dictionary, split=split_against_dictionary)


def test():
    text_batch_simulation = [["This is a long article with \"quotations,\" punctuation marks, spaces, and periods. Don't forget to use lowercase on nouns that don't need it!"],
                            ["We need a second dimension here because for a string input we have to specify the shape to be (1,) which is a one-dimensional vector with only one element."],
                            ["Here is some math to throw a wrench into things: \"4+4=8 so long as y=mx+b.\""]]
    input_layer = layers.Input(shape=(1,), dtype=tf.string) 

    tv_output = tv(input_layer)

    tv_demo = keras.Model(input_layer, tv_output)

    tv_demo.summary()
    tv_demo.compile()
    # A test to show that the model can convert string inputs into integer indicies. 
    print(tv_demo.predict(text_batch_simulation[0]))


# Find mean and standard deviation of the indices of the dictionary
dict_indices = numpy.array([index for index in range(1, len(words)+1)])
dict_mean = numpy.sum(dict_indices)/len(dict_indices)
dict_variance = numpy.sqrt(numpy.sum((dict_indices - dict_mean)**2)/len(dict_indices))


# Create a little custom function to cast the integer data into floats and normalize the data
@keras.utils.register_keras_serializable(package="WebScraper")
def normalize(x):
    return (tf.cast(x, tf.float32) - dict_mean)/dict_variance


# Create a little custom function to undo the normalization and recast the data as integers
@keras.utils.register_keras_serializable(package="WebScraper")
def reverse_normalization(x):
    return tf.cast((x * dict_variance) + dict_mean, tf.int32)


def define_disc_prep():
    # Create the input layer of the discriminator which is technically the same shape as the generator
    # The shape is one string
    input_layer = layers.Input(shape=(1,), dtype=tf.string) 

    disc = tv(input_layer)

    return keras.Model(input_layer, disc)


def define_discriminator():
    # Create the input layer of the discriminator which is technically the same shape as the generator
    # The shape is 1D of an unknown number of ints.
    input_layer = layers.Input(shape=(None,), dtype=tf.int32)

    # Convert int indexes to float values
    # Normalize data to small values so we don't have massive ints floating aroud the system.
    disc = layers.Activation(normalize)(input_layer)

    # Shape data to fit LSTM
    disc = layers.Reshape((-1, 1))(disc)

    # Add a LSTM layer
    # Tanh is automatically applied to LSTM as an activation function
    disc = layers.Bidirectional(layers.LSTM(1024))(disc) 
    disc = layers.BatchNormalization()(disc)
    disc = layers.Dropout(.5)(disc)

    disc = layers.Dense(256)(disc)
    disc = layers.LeakyReLU()(disc)
    disc = layers.BatchNormalization()(disc)

    disc_output = layers.Dense(1, activation="sigmoid")(disc)

    discriminator = keras.Model(input_layer, disc_output)

    discriminator.summary()

    discriminator.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(), metrics='accuracy')

    return discriminator



def define_generator():
    # Create the input layer of the discriminator which is technically the same shape as the generator
    # The shape is one string
    input_layer = layers.Input(shape=(1,), dtype=tf.string) 

    gen = tv(input_layer)

    # Convert int indexes to float values
    # Normalize data to small values so we don't have massive ints floating aroud the system.
    gen = layers.Activation(normalize)(gen)

    # Shape data to fit LSTM
    gen = layers.Reshape((-1, 1))(gen)

    # Add a LSTM layer
    # Tanh is automatically applied to LSTM as an activation function
    gen = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(gen) 
    gen = layers.BatchNormalization()(gen)
    gen = layers.Dropout(.5)(gen)

    gen = layers.Bidirectional(layers.LSTM(512))(gen) 
    gen = layers.BatchNormalization()(gen)
    gen = layers.Dropout(.5)(gen)

    gen = layers.Dense(512)(gen)
    gen = layers.LeakyReLU()(gen)
    gen = layers.BatchNormalization()(gen)
    gen = layers.Dropout(.4)(gen)


    # Create 100 dense nodes that will each be outputs that are also sequentially added to the next node as an input.
    final_layers = [layers.Dense(1, activation="tanh") for layer in range(100)]
    output_nodes = []
    for i, dense in enumerate(final_layers):
        if i>0:
            prev_nodes = [output_nodes[node] for node in range(i)]

            # Append the last general layer of the generator for easy concatenation.
            prev_nodes.append(gen)
            concat = layers.Concatenate()(prev_nodes)
            output_nodes.append(final_layers[i](concat))
        else:
            # If this is the first run through, just attach gen to the output
            output_nodes.append(final_layers[i](gen))


    # Concatenates all the output nodes together for easy transformation.
    final_concat = layers.Concatenate()(output_nodes)
    
    # Transform each output into an index of the dictionary.
    gen_ints = layers.Activation(reverse_normalization)(final_concat)

    generator = keras.Model(input_layer, gen_ints)
    return generator


def define_GAN(discriminator, generator):
    # make weights in the discriminator not trainable
	discriminator.trainable = False
	# get article input from generator model
	article = generator.input
	# get summary output from generator
	gen_output = generator.output
	# connect summary output from generator as input to discriminator
	gan_output = discriminator(gen_output)
	# define gan model as taking article and outputting a classification
	gan = keras.Model(article, gan_output)
	# compile model
	opt = optimizers.Adam(lr=0.0002, beta_1=0.8)
	gan.compile(loss='binary_crossentropy', optimizer=opt)
	return gan


# Pick data randomly from the dataset
def generate_real_samples(disc_prep, summaries, n_samples):
    # Pick random indices
    indices = numpy.random.randint(0, summaries.shape[0], n_samples)

    # Get randomly selected pairs
    xdata = summaries[indices]

    # Transform summaries into discriminator-readable data
    xdata = disc_prep.predict(xdata)

    # Generate real labels (true)
    labels = numpy.ones((n_samples, 1))
    return xdata, labels


def generate_fake_samples(gen, articles, n_samples):
    # Pick random indices
    indices = numpy.random.randint(0, articles.shape[0], n_samples)

    # Get random articles
    xdata = articles[indices]

    # Generate summaries based on those articles
    summaries = gen.predict(xdata)

    # Generate fake labels (false)
    labels = numpy.zeros((n_samples, 1))

    return summaries, labels


# NOTE: Add latent-dims if we want the summarization to be a bit randomized (not the same every time).
def train(generator, disc_prep, discriminator, gan, dataset, n_epochs=100, n_batch=128): 
    import tqdm
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in tqdm(range(n_epochs)):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected real summaries
            real_summaries, real_labels = generate_real_samples(disc_prep, dataset[1], half_batch)

            # train discriminator model weights on real samples
            d_loss_real, _ = discriminator.train_on_batch(real_summaries, real_labels)

            # generate 'fake' examples
            fake_summaries, fake_labels = generate_fake_samples(generator, dataset[0], half_batch)

            # train discriminator model weights on fake samples
            d_loss_fake, _ = discriminator.train_on_batch(fake_summaries, fake_labels)

            # prepare random articles to update generator loss
            # Pick random indices
            indices = numpy.random.randint(0, dataset[0].shape[0], n_batch)

            # Get random articles
            zdata = dataset[0][indices]

            # create inverted labels for the fake samples 
            # If discriminator is wrong (returns false) then the loss of the generator decreases
            # If we did not invert the label, the generator would think it was wrong every time it was right.
            # A generator win, is a discriminator loss and vice versa.
            inverted_labels = numpy.ones((n_batch, 1))

            # update the generator via the discriminator's error
            g_loss = gan.train_on_batch(zdata, inverted_labels)
            # summarize loss on this batch
            print('>%d, %d/%d, dreal=%.3f, dfake=%.3f gen=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))



def main():
    # Load dataset (pairs of articles and summaries from wikipedia)


    # If this file is being run as the main, define the GAN models and train them.
    disc_prep = define_disc_prep()
    discriminator = define_discriminator()
    generator = define_generator()
    gan = define_GAN(discriminator, generator)

    # generator.summary()
    # gan.summary()

    # discriminator.save("./Discriminator")
    # generator.save("./Generator")

    #train(generator, disc_prep, discriminator, gan, dataset, n_epochs=20, n_batch=128)



if __name__ == "__main__":
    main()
    #test()