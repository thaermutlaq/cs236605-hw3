r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT"
    temperature = 0.1
    # ========================
    return start_seq, temperature


part1_q1 = r"""
When we have very long sequences like whole text, RNNs can face the problem of vanishing gradients.
When attempting to back-propagate across very long input sequences may result in vanishing gradients, and in turn, an unlearnable model.

In addition, long sequences may result in the problem of very long training times.
"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
Since we want our module to be able to generate full text and not only sequence,
we should be implementing a stateful module and not a stateless one. Therofore, the order of the sequence is importnat and 
the hidden state is propogated between the batches in the same epoch.
In this case the sequence memory will persists across sequences and if sequence B is fed after sequence A, we want the network
to evaluate sequence B with memory of what was in sequence A. Therfore the order of the train data is important to be able to generate long sequences 
even when we train on small sequence length.
"""

part1_q4 = r"""
1. The smaller the temprature is the larger the values that we apply the softmax on.
Performing softmax on larger values makes the RNN network more confident and more conservative in its samples 
(less input is needed to activate the output layer & less likely to sample from unlikely candidates).
This will cause the network to generate "safe" guesses,  which is good for sampling.

In the other hand, using a higher temperature produces a softer probability distribution over the classes,
resulting in more diversity.
This will cause the network start generating "riskier" guesses, which is better for trainning.

2.When the temperature is close to 1 we can see a lot of mistakes in words (words without meaning). Words being sampled
with strange letters. 
When the temperature is very high (10 for example), we get a gebrish text with no words and no sentences, completely random letters. 

This is caused because the network try to sample "creative" and "riskier" guesses according to the expilination in 1. 
this cause the network to sample incoorect words instead using known ones.


3.When we use a very small temperature, we can see that the network repeat common words like (I, will, not the). 
Every sentence start in the same words, and no "strange" and "hard" words are used only primitive words that is common.

This is caused because the network try to sample in a "safe" manner, therefore it samples a lot of known words. 
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 1024
    hypers['z_dim'] = 128
    hypers['learn_rate'] = 0.001
    hypers['x_sigma2'] = 0.001
    hypers['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
The hyperparameter $\sigma^2$ controls the division ratios and calibrate the data loss and KLD loss in the total loss, 
which will be translated to a variance in the generated photos by the module. 
The higher the $\sigma^2$ the smaller the data loss part in the loss function and the KLD loss will be more dominant,
this will cause the module to be more calibrated to the latent space part and generate similar photos (like an average photo).
However, when the $\sigma^2$ is smaller, the data loss part in the total loss is bigger, the module will try to give more focus
on the "original" photo that were encoded to the random z that we generated, and since vectors are sampled randomly, diffrent samples
will cause diffrent decoded photos with a variance.
"""

part2_q2 = r"""
In the VAE loss term, we sum up two separate losses: 
1. The generative loss, which is a mean squared error that measures how accurately the network reconstructed the images.
This term encourages the decoder to learn to reconstruct the data. If the decoder’s output does not reconstruct the data well, 
statistically we say that the decoder parameterizes a likelihood distribution that does not place much probability mass on the true data.

2. A latent loss, which is the KL divergence that measures how closely the latent variables match a unit gaussian.
We can think about it like regularization term. This is the Kullback-Leibler divergence between the encoder’s distribution 
(posterior distribution of points in the latent spaces given a specific instance) $p(\bb{Z}|\bb{X})$ and the **prior** distribution $p(\bb{Z})$
This divergence measures how much information is lost, when using q to represent p. It is one measure of how close q is to p.

Since we sample in the latent space with a normally distributed gaussain $\bb{u}\sim\mathcal{N}(\bb{0},\bb{I})$ and then apply the **reparametrization trick**
The KL divergence help to optimize the distribution of X so that they are more tightly packed around the origin. 
So we are going to optimize so that the P distribution look the most like the N(0,1) distribution (a gaussian distribution located around the origin).
This have a big benefits that the samples that we take in the latent space from the normal distribution will be likelly mapped 
by the decoder to an close image to what we trained on.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 1
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002

    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


