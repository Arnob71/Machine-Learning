import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *

def costFunction(X, W, b, Y, R, lm):
    J = (tf.linalg.matmul(W,tf.transpose(X))+tf.transpose(b)-tf.transpose(Y))*tf.transpose(R)
    J = 0.5*tf.reduce_sum(J**2)+(lm/2)*(tf.reduce_sum(X**2)+tf.reduce_sum(W**2))

    return J

X, W, b, numOfMovies, numOfFeatures, numOfUsers = load_precalc_params_small()
Y, R = load_ratings_small()

myRatings = np.zeros(numOfMovies)

myRatings[2700] = 5 

#Or suppose you did not enjoy Persuasion (2007), you can set
myRatings[2609] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
myRatings[929]  = 5   # Lord of the Rings: The Return of the King, The
myRatings[246]  = 5   # Shrek (2001)
myRatings[2716] = 3   # Inception
myRatings[1150] = 5   # Incredibles, The (2004)
myRatings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
myRatings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
myRatings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
myRatings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
myRatings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
myRatings[2937] = 1   # Nothing to Declare (Rien à déclarer)
myRatings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
myRated = [i for i in range(len(myRatings)) if myRatings[i] > 0]

Y = np.c_[myRatings, Y]
R = np.c_[(myRatings != 0).astype(int), R]
yNorm, yMean = normalizeRatings(Y,R)

numOfMovies, numOfUsers = Y.shape
numOfFeatures = 100

tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((numOfUsers, numOfFeatures),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((numOfMovies, numOfFeatures),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          numOfUsers),   dtype=tf.float64),  name='b')

opt = keras.optimizers.Adam(learning_rate=1e-1)

iters = 500

for i in range(iters):
    with tf.GradientTape() as tape:
        cost = costFunction(X,W,b,yNorm,R,1)
    grads = tape.gradient(cost,[X,W,b])
    opt.apply_gradients(zip(grads,[X,W,b]))
    if i%20==0:
        print(f"{i}:{cost}")

prediction = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
pm = prediction+yMean
myPred = pm[:,0]
ix = tf.argsort(pm[:,0], direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in myRated:
        print(f'Predicting rating {myPred[j]:0.2f}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(myRatings)):
    if myRatings[i] > 0:
        print(f'Original {myRatings[i]}, Predicted {myPred[i]:0.2f}')




