'''
---> Hidden Markov Model (Reinforcement Learning)

The Hidden Markov Model is a finite set of states, each of which is associated with a (generally multidimensional) probability distribution [].
Transitions among the states are governed by a set of probabilities called transition probabilities.
In a particular state an outcome or observation can be generated, according to the associated probability distribution.
It is only the outcome, not the state visible to an external observer and therefore states are ``hidden'' to the outside; hence the name Hidden Markov Model.
'''
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.2, 0.8]])  # refer to points 3 and 4 above
# the loc argument represents the mean and the scale is the standard devitation
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()
print(mean)

# in the new version of tensorflow we use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())