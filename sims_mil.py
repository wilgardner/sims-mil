#-------------------------- Tensorflow implementation of MIL for MSI data ---------------------------#

# version 1.1 - February 2024
# Created by WIL GARDNER
# Centre for Materials and Surface Science
# La Trobe University

#----------------------------------------------------------------------------------------------------#

import tensorflow as tf
from tensorflow.keras import layers

def create_mil_model(input_dim, gamma, fe_units=[]):
  
    inputs = layers.Input(shape=(input_dim,))
    h = inputs

    # Initial feature extractor
    for i, u in enumerate(fe_units):
        if i == len(fe_units)-1:
            act = 'softmax'     #Features 
        else:
            act = 'relu'        #Hidden layers
        h = layers.Dense(units=u, activation=act)(h)
    
    # Weight vectors/matrices:
    Wq = layers.Dense(units=h.shape[-1], name='Wq')
    Wv = layers.Dense(units=h.shape[-1], name='Wv')
    W0 = layers.Dense(units=1, name='W0')
    Wb = layers.Dense(units=1, name='Wb')

    # First stream
    scores = W0(h)

    # Critical instance embeddings
    critical_index = tf.argmax(scores, axis=0)

    # Max score
    max_score = tf.gather(scores, critical_index)
    
    # Instance embeddings transformation
    q = Wq(h)
    v = Wv(h)
    
    # Critical instance embeddings
    hm = tf.gather(h, critical_index)
    qm = Wq(hm)

    # Distance measurement U
    distances = tf.reduce_sum(tf.multiply(q, qm), axis=-1)
    U = tf.nn.softmax(distances, axis=0)
    U_expanded = tf.expand_dims(U, axis=-1)

    # Bag embedding
    Uv = tf.multiply(U_expanded, v)
    b = tf.reduce_sum(Uv, axis=0)

    # Bag score
    b_expanded = tf.expand_dims(b, axis=0)
    bag_score = Wb(b_expanded)
    
    # Final score    
    final_score = 0.5 * (bag_score + max_score)
    score_output = tf.keras.layers.Activation('sigmoid', name='score_output')(final_score)

    model = tf.keras.Model(inputs=[inputs], outputs=[score_output, h, U])
        
    # Entropy regularisation
    U_entropy = -tf.reduce_sum(U * tf.math.log(tf.clip_by_value(U, 1e-12, tf.float32.max)), axis=0)*(1/tf.math.log(tf.cast(tf.shape(U)[0], tf.float32)))
    model.add_loss(-gamma*U_entropy)
    return model