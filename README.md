
# NN Training Dynamics: Making Sense of a Confusing Mess

## Table of Contents
0. [TL;DR](#0-tldr)
1. [Introduction: 4 mysteries of training NNs](#1-intro)
2. [A simple Quadratic Stochastic Model](#2-quad-model)
3. [The Hessian Spectrum Throughout Training](#3-hessian-training)
4. [Testing the Quadratic Approximation](#4-testing-quad)
5. [Is there Structure in the Eigenvectors components?](#5-eigvec-structure)
6. [Are the Eigenvectors Mostly Constant?](#6-eigvec-constant)
7. [A plausible hypothesis: Narrowing Valleys](#7-valley-hypothesis)
8. [Testing the Narrowing Valley Hypothesis](#8-testing-valley)
9. [Conclusion and Further Questions](#9-conclusion)

## 0. TL;DR <a name="0-tldr"></a>

In this blog post I attempt to answer the following questions:

1. why does lowering the learning rate during training result in a sharp drop in error?
2. why is increasing the batch size equivalent to lowering the learning rate? 
3. why does the largest eigenvalue  of the hessian increase as learning rate drops? 
4. why do negative eigenvalues persist in the hessian throughout  training?

It turns out that thinking of the loss landscape of neural networks as a long valley with increasingly tightening walls (see picture below) which is stochastically translated (but not stretched) in random directions at every minibatch can elegantly answer these 4 questions.

I replicate all 4 effects above in the context of a small resnet trained on cifar10 and provide evidence for the "stochastic narrowing valley" hypothesis by first solving the dynamics of SGD-with-momentum for a stochastic quadratic function, then by computing the eigenspectrum of the hessian of the resnet at many training checkpoints.

![curving_valley_1.png](images%2Fcurving_valley_1.png)

## 1. Introduction: 4 mysteries of training NNs <a name="1-intro"></a>

This is a blog post about investigating a bunch of weird behaviours that happen when training modern neural networks. Often, newcomers to deep learning look at the size of current models, observe that the optimization problem is non-convex in dimension 10^8, and throw up their hands in despair, abandoning any attempt at building intuitions about the loss landscape of realistic networks. As we'll see, that despair might be a bit premature, as we can build very simple 1D or 2D loss landscapes that capture a lot of the properties we observe in modern deep learning, while still being very easy to visualize. 

That being said, here are 4 puzzling things that happen when training networks:

1. **Sharp Loss Decrease at Learning Rate decreases**: we observe a sharp, cliff-drop drop in loss every time we suddenly decrease the learning rate.
2. **Equivalence Between High Batch Size and Low Learning Rate**: Increasing the batch size and decreasing learning rates have the same effect on the loss.
3. **Edge of Stability**: The highest eigenvalue of the hessian rises precisely to match the maximum value allowed by the current learning rate.
4. **Persistent Negative Eigenvalues**: The negative eigenvalues of the hessian persist throughout training, they don't get optimised away.

Somewhat surprisingly, it turns out that these 4 separate phenomena can be unified into a simple coherent picture of what's going on. But first, let's think a bit about why each of these mysteries is at odds with the simple Calculus 1 "gradient descent to a local minimum" picture of what's going on. 

**Mystery #1**: In the Cal1 view, the learning rate is roughly "how big of a step" we're taking down the mountain, if we suddenly reduce our step size, it makes no sense for the loss to suddenly go down a cliff, we'd expect to just make much slower progress down the mountain.

**Mystery #2**: Of the four effects above, this is the easiest for the Cal1 view to accomodate, increasing batch size lowers the variance of your gradient estimate, and it makes sense that you take careful, small steps down the mountain if you are uncertain about the current slope. Yet the exact quantitative equivalence is harder to explain. Multiplying the batch size by 5 has almost the exact same effect as dividing the learning rate by 5. 

**Mystery #3**: A decrease in learning rate somehow makes the network go towards a region of the landscape that is as sharp as the current learning rate would allow before the optimization starts diverging. What's keeping the eigenvalue from ballooning up even higher? Is the loss landscape somehow fractal in nature? And every time we decrease the learning rate we drop down into a pit that we were previously skipping over? 

**Mystery #4**: Ordinarily the negative eigenvalues are the easiest to optimise since they are the most unstable: a step step down the slope *increases* your gradient magnitude towards the minimum. Yet as we'll see, a sizeable fraction of the eigenvalue spectrum at every point in training consists of negative values. What's keeping them from being optimised away?

## 2. A simple Quadratic Stochastic Model <a name="2-quad-model"></a>

The simplest model that exhibits the strange cliff-drop-at-lr-decrease feature is optimising a quadratic function $f(x) = \lambda (x-\epsilon)^2$, where $\epsilon \sim N(0, \sigma^2)$ is a random variable added every time the function is sampled. When doing gradient descent with a learning rate $\alpha$ on such a stochastic function, the update equations become:

$$f(x) = \lambda (x-\epsilon)^2$$ 

$$ f'(x) = 2\lambda (x-\epsilon)$$ 

$$ x_{n+1} = x_n - 2\alpha \lambda (x_n-\epsilon) $$

$$x_{n+1} = x_n - 2\alpha \lambda x_n + \epsilon^*$$

$$ x_{n+1} = x_n\bigg(1-2\alpha\lambda\bigg) + \epsilon^* $$


Where $\epsilon^* \sim N\big(0, (2\alpha\lambda\sigma)^2\big) = N(0, \eta^2)$, introducing a new variable $\eta$ for later convenience. Now, the randomness injected via $\epsilon^*$ will be counterbalanced by the shrinking with factor $\bigg(1-2\alpha\lambda\bigg)$, and at equilibrium we'd expect both effects to counterbalance each other. If we assume that at equilibrium each $x_n$ to be independent of $x_{n-1}$ and distributed according to a simple gaussian $p(x) = N(0, s^2)$, we can solve for $s^2$ and obtain the following result:

$$ s^2 = \frac{\alpha\lambda\sigma^2}{1-\alpha\lambda}$$

This corresponds to the variance of $x$ as it is bumped around the minimum of the quadratic by noise. And hence that the expected value of the function $\lambda x^2$ at equilibrium is

$$ E\[\lambda x^2\] = \lambda s^2 = \frac{\alpha\lambda^2\sigma^2}{1-\alpha\lambda}$$

This is telling us that SGD on such a stochastic function drops down until a level roughly proportional to both the learning rate $\alpha$ and the variance in the minimum of the function $\sigma^2$. Decrease either $\alpha$ or $\sigma^2$ by a factor of 10 is expected to decrease the loss by the same factor. 

Hence this simple model seems to exhibit both Mystery #1 and Mystery #2. Learning rate drops are always accompanied by sharp drops in loss, as the optimisation settles to a new equilibrium. And there is an almost exact equivalence between dropping the learning rate and dropping the variance of the stochastic term (the analog to increasing the batch size).

If we directly plot the loss of such a model, periodically dropping the learning rate by a factor of 10, it looks something like this:

![stochastic_quad_lr_decrease.png](images%2Fstochastic_quad_lr_decrease.png)

Notice the log scale on the y axis, and the fact that each new minimum level is exactly an order of magnitude below the previous one, exactly as predicted by the theory.

As an aside, we can derive (with just slightly more effort, see the [appendix](#sgd-mom-derivation)) an analogous equation for the case of SGD-with-momentum, where $\gamma$ is the momentum term. For this we obtain the following expected loss at equilibrium:

$$ E\[\lambda x^2\] = \frac{\alpha\lambda^2\sigma^2}{1-\gamma^2 - \alpha\lambda} $$

Which of course reduces to the non-momentum case when $\gamma=0$. Somewhat surprisingly, momentum *hurts* the equilibrium loss level, it allows us to achieve equilibrium earlier, but once there, it tends to push us to a higher level than pure sgd. Note that the formula above breaks down when our learning rate and momentum is high enough that we'd get oscillatory behavior even without noise.

We can test the formula above empirically on the test quadratic surface, and see that it accurately predicts the loss level: 

![sgd_with_momentum.png](images%2Fsgd_with_momentum.png)

The two curves have the same learning rate, they only differ in the momentum term. Notice the faster initial slope of sgd-with-momentum at the cost of a higher equilibrium loss level.

### Extension to multiple dimensions

Extending this effect to n dimensions is straightforward, a generic positive definite quadratic function can be written as $f(x) = (x-\mu)^T H (x-\mu)$. We can diagonalize $H$ to get $H = Q^T \Lambda Q$ with diagonal $\Lambda$ and orthogonal $Q$. Then $f(x) = (x-\mu)^T Q^T \Lambda Q (x-\mu)$ and the simple linear change of variables $y = Q (x-\mu)$ turns the function into $f(x) = y^T \Lambda y = \sum_i \lambda_i y_i^2$, which is a simple sum of independent quadratics, each of which can have its minimum stochastically vary as above.

### Non-Equilibirum Behavior

Having found the behavior of SGD on our simple stochastic quadratic loss in the limit of equilibrium, we now ask what happens out of equilibrium. In this toy model, we will assume that noise is essentially irrelevant until $x$ reaches values close to $0$. More specifically we should expect noise to become relevant when $x \sim s$, i.e. when it becomes similar in size to the standard deviation of the empirical equilibirum distribution. We can compute the relaxation time $\tau$ that it takes SGD to get to equilibrium:


$$ x_{n+1} = x_n - 2\alpha\lambda x_n = x_n (1-2\alpha\lambda) \implies x_n = x_0 \bigg(1-2\alpha\lambda\bigg)^n $$

$$s = x_0 \bigg(1-2\alpha\lambda\bigg)^\tau \implies \tau = \frac{\log (s/x_0)}{\log (1-2\alpha\lambda)}$$

$$\tau = \frac{\frac{1}{2}\log \bigg(\frac{\alpha\lambda\sigma^2}{x_0(1-\alpha\lambda)}\bigg)}{\log (1-2\alpha\lambda)}$$

We therefore have an expression for the expected number of iterations it takes for SGD to descend down to a level where its dynamics are dominated entirely by the noise.

### What's the point of momentum? Why not just use a larger learning rate?

On quadratic functions for which $\alpha\lambda << 1$, i.e. those functions with small enough eigenvalues that gradient descent is far from divergence, adding momentum is mostly equivalent to simply multiplying the learning rate by $1/(1-\gamma)$. But for functions where GD is close to oscillatory behavior, adding momentum *does not* cause divergence, like using a larger learning rate would.

Hence the real impact of momentum is to allow the very small eigenvalues to have effective learning rates of $\alpha/(1-\gamma)$ while still preventing divergence on the large eigenvalues of our function.

### Relationship to Bayesian Posterior Sampling, why does noise help generalisation?

If we imagine sampling the parameters from their posterior distribution, we would expect the variance in each eigendirection to be $1/s^2 \propto \lambda$, because we'd approximately be sampling from a MVN with covariance given by the Hessian $H = \Sigma^{-1}$. Yet the equilibrium variance we obtain from SGD is $s^2 = \frac{\alpha\lambda\sigma^2}{1-\alpha\lambda}$. We see that the $\lambda$ dependence has the wrong form, we are missing a factor proportional to $1/\lambda^2$ in order for the stochasticity caused by SGD to simulate sampling from the posterior. We can make the SGD equilibrium noise closer to bayesian sampling by making $\sigma^2$ and/or the learning rate $\alpha$ depend on eigenvalue, in both cases we need higher values of $\sigma^2$ and $\alpha$ at low $\lambda$.

This suggests a mechanism through which different optimisation algorithms and noise injection schemes might be helping generalisation: they're changing $\sigma^2$ and $\alpha$ in each eigendirection in a way that makes the equilibrium distribution better match the posterior distribution.


### What does the Simple Quadratic Model teach us?

The fundamental lesson of this simple model is likely that **the noise in our optimisation function** is a crucial factor to keep in mind when thinking about loss landscapes. Mysteries #1 and #2 above are fundamentally noisy phenomena. This toy model is also evidence against the landscape being somehow fractal in nature given that we don't need to invoke such a complicated structure to explain the sharp loss decreases.

## 3. The Hessian Spectrum Throughout Training <a name="3-hessian-training"></a>

### Model Setup: Small Resnet on Cifar10

Now that we've derived a plausible model for what's happening in the loss landscape, let's investigate the landscape of a real neural network by explicitely computing the full Hessian at multiple points in training. Here's the setup for the experiment:

- CIFAR10 dataset without data augmentation
- Very Tiny Resnet model with GELU activations (for twice differentiability), only 26000 parameters in total
- SGD with momentum. lr=1e-1, momentum=0.97, weight-decay=1e-3
- 500 epoch total training
- lr decreases by 10 at iterations = 10000, 20000, 30000, 40000
- batch size 512
- final accuracy of 80% on validation set

Computing the full dataset Hessian is only feasible for very small models, which is why we choose such a small resnet.

Here's what the minibatch loss looks like over time:

### Training graphs

![training_graph.png](images%2Ftraining_graph.png)

Notice the sharp drops at iterations 10000 and 20000, corresponding to dividing the learning rate by 10. Now let's take every checkpointed network and compute its total loss on the training set, as well as the largest eigenvalue of the hessian of the full training set at that point:

![loss_vs_max_eigval_over_training.png](images%2Floss_vs_max_eigval_over_training.png)

The loss cliff drops become much cleaner, and we can see an extra drop appear at iteration 30000 and a very small one at 40000. There's also clear evidence of the "edge of stability" phenomenon: the top eigenvalue keeps increasing throughout training, and it shoots up quickly after each learning rate decrease.

### Hessian Eigenvalue Spectrum Evolution


Now let's look at the full spectrum of the Hessian, and how it evolves through training. In the figure below we're plotting a log-log graph of the sorted eigenvalues at various points in training, denoted by the iteration number.

![pos_neg_eigvals_through_training.png](images%2Fpos_neg_eigvals_through_training.png)

A few observations:

- The positive eigenvalues have a roughly power-law shape to them. The 100-th biggest value is roughly 100 times smaller than the top value. There are many small eigenvalues and few top eigenvalues, but the shape of the spectrum is such that the total power at every scale is roughly the same.
- The positive spectrum mostly keeps the same overall shape through training, except for an overall translation upwards, which corresponds to the Edge-of-Stability effect. 
- We still have negative eigenvalues even at the very end of training, though the shape of the spectrum does seem to change, and the total number of non-negligeable negative values steadily drops (the dropoff in the value with rank shifts leftward as training progresses)
- The pure quadratic assumption is already violated by the changing spectrum (and by the existence of negative eigenvalues).

## 4. Testing the Quadratic Approximation <a name="4-testing-quad"></a>


In the derivation of the stochastic quadratic model, we made a really big assumption about the form of the stochasticity, namely that the function's shape stays the same, and only its minimum is shifted randomly from sample to sample. One could imagine other forms of stochasticity, for instance, the sharpness of the minimum might also change in addition to its minimum location. Or the minimum location might vary in a non-gaussian way. Here we test this assumption for our network at iteration = 10000, i.e. right before the first decrease in lr (though these results replicate at every other point in training).

### The positive eigenvalue directions

To test this assumption, for each eigenvector $v_\lambda$, sample lots of minibatches of size 512 from the training set, and plot the line searches for each minibatch, i.e. we plot $f_i(t) = f_i(\theta + t v_\lambda)$. The figure below is showing $f_i(t)$ (with scaled units for t in order to show non-trivial behavior) for a particular eigenvalue (all positive eigenvalues behave essentially the same). The minimums of each curve are shown with blue dots. The bottom graph plots the derivative of the top curves, exact quadratics should have linear derivatives, and this is what we get.

![positive_eigen_line_search.png](images%2Fpositive_eigen_line_search.png)

A few observations:

- The sharpness of the function doesn't change minibatch to minibatch, they're all basically the same shape, up to an irrelevant translation upwards
- The minimum locations do seem to be distributed normally, no weird surprises or outliers here.
- Overall the toy stochastic quadratic model seems to perfectly describe what's going on here.

### The negative eigenvalue directions

The positive eigenvalue directions behave as the toy model expected, but what about the negative directions? Doing the same procedure as above, this is what we get, again plotting line searches in a particular eigenvector, with different curves representing different minibatches. Blue points are the global minimums of the functions.

![negative_eigen_line_search.png](images%2Fnegative_eigen_line_search.png)

- Again the function shape remains fairly consistent batch-to-batch, almost every function has two local minima, and they all cluster around the same two points on the x-axis.
- one of the two local minima is clearly lower than the other, but not all minibatches agree on which of the two is the correct one.
- The most surprising fact here is that these directions have not yet been optimised away. In these plots the middle point represents the unchanged parameters of the network, i.e $f_i(\theta + 0 v_\lambda)$, and we see that this point lies at a local maximum of the function. These negative eigenvalues are also quite large, it's not the case that this direction is just too flat for SGD to make progress. **Some unknown mechanism** is keeping the network at a local maximum in this direction.

### How does the variance of the minimum location vary with eigenvalue?

The toy quadratic model has a free parameter $\sigma^2$, the variance of the minimum location computed across minibatches. A priori this variance is free to depend on the eigenvalue $\lambda$, so let's plot $\sigma$ against the positive $\lambda$:

![min_std_vs_eigenvalue.png](images%2Fmin_std_vs_eigenvalue.png)

So we see a decrease in standard deviation for larger eigenvalues, but notice the log scale on the x-axis: an order of magnitude increase in eigenvalue gives us a measly $\sim 0.3$ decrease in standard deviation. The low eigenvalues have much less noise than we might have expected of them.

### How does the equilibrium std-dev $s$ vary with eigenvalue?

Now for each eigenvalue we plug in the relevant factors into the equation for $s^2$ in order to get the typical size of fluctuations at equilibrium, assuming lr=0.1, which was its value from iterations 0 to 10000 (the point in training at which we compute all these quantities)

$$ s^2 = \frac{\alpha\lambda\sigma^2}{1-\alpha\lambda}$$

![s_vs_eigenvalue.png](images%2Fs_vs_eigenvalue.png)

Perhaps surprisingly, higher eigenvalues tend to oscillate slightly more at equilibirum (apart from a few outliers at the very high end). But again the log scale on the x-axis implies that there's remarkably little change for the wide range of values that the eigenvalues pass through.

### How Long Should the positive eigenvalues take to reach equilibrium?

Given the variance $\sigma^2$ in the function, the eigenvalue $\lambda$, the learning rate $\alpha=0.1$ and the initial distance from equilibrium $x_0$, we can compute the time-to-equilibrium with the formula from section 2:

$$\tau = \frac{\frac{1}{2}\log \bigg(\frac{\alpha\lambda\sigma^2}{x_0(1-\alpha\lambda)}\bigg)}{\log (1-2\alpha\lambda)}$$

We now plot this $\tau$ against the positive eigenvalues of our network (we compute $x_0$ for each corresponding eigenvalue by projecting the total parameter difference between iteration 0 and iteration 10000 onto the eigenvectors).

![time_to_equilibrium_vs_eigenvalue.png](images%2Ftime_to_equilibrium_vs_eigenvalue.png)

Notice the log scales on both the y and x axis this time. The very highest eigenvalues take almost no time at all to be optimised, whereas the lowest ones take as much as $10000$ iterations. A lot of assumptions went into this graph though, chiefly that the spectrum is constant from iteration 0 to iteration 10000, which is not true. This also doesn't take momentum into account, which would roughly multiply the learning rate by a factor of $1/(1-\gamma)$, and hence translate the whole graph downward by roughly that amount. For the network we've training, $\gamma=0.97$, hence $1/(1-\gamma) \approx 33$, which means that the lowest eigenvalues on the graph should be translated downward by a signficant amount. 

This graph gives us a plausible hypothesis for why we need to train a high learning rates: the low eigenvalues take a long time to reach equilibrium. Therefore we can see the tradeoff between low lr and high lr as follows:

- We need *high lr* in order to reduce the time it takes to reach equilibirum in the low eigenvalue directions.
- But we need *low lr* in order to decrease variance of the oscillations at equilibrium, and reach a lower loss level.

If we lower the learning rate too quickly, we'll get a sudden drop in loss as all the eigenvalues that were already at equilibrium drop to an even lower level, but the price we pay is that those directions that weren't yet at equilibrium now will take much longer to get there, because the learning rate is much smaller, and so we've crippled our long-term potential.

Once all directions have reached equilibrium, no more progress is possible at this learning rate, and we *need* to lower the learning rate or increase the batch size to make any further progress. **However**, as we'll see in section 7, this effect is not enough to completely explain why small learning rates are important, it does seem to explain some of the effect, but only a fraction of it.

### How much loss decrease does the toy model predict?

Now that we have the variance $s^2$ (the variance of the equilibrium distribution) for each eigenvalue, we can multiply it by $\lambda$ to get the expected equilibrium loss in that direction. $E\[\lambda x^2\] = \lambda E\[x^2\] = \lambda s^2$. This quantity also tells us the potential size of the drop in loss we could get if noise went all the way to zero, so by summing this expected loss across all positive eigenvalue directions, we obtain a prediction for the total loss drop of the network. In the following graph we plot the cumulative sum of the expected losses from the smallest eigenvalue up to the eigenvalue on the x-axis. The red line corresponds to the total loss of the network at iteration 10000 (the checkpoint at which we compute everything).

![expected_cumul_loss_vs_eigenvalue.png](images%2Fexpected_cumul_loss_vs_eigenvalue.png)

As we can see, something goes terribly wrong: the toy model is predicting a total loss decrease an order of magnitude higher than the total loss of the network. Cross-Entropy is bounded below by 0, hence it's impossible to get a loss decrease larger than the red line. I don't know what's going on here, perhaps the addition of gradient clipping and weight decay is messing up the simple sgd-with-momentum math, or the noise is non-gaussian in a way that makes our assumptions break down. (The noise merely being correlated between dimensions wouldn't be enough to explain this.)


## 5. Is there Structure in the Eigenvectors components? <a name="5-eigvec-structure"></a>
Switching gears a bit for the moment, let's look at the eigenvectors corresponding to each of our eigenvalues and try to figure out if there's any internal structure to them that we can find. An eigenvector $v_\lambda$ will by construction have unit norm, i.e. $\sum_i v_{\lambda, i}^2 = 1$, but let's figure out how sharply concentrated those $v_{\lambda, i}^2$ values are. Do we have a few components that dominate the norm, or is it pretty much indistinguishable from a gaussian normal vector? 

To do this, for each vector $v_{\lambda}$, sort the values $v_{\lambda, i}^2$ from highest to lowest into $v_{\text{sorted},\lambda, i}^2$, and compute the cumulative sum $\sum^n_i  v_{\text{sorted},\lambda, i}^2$. This is what we plot below, with red curves corresponding to high $\lambda$ vectors, and bluer curves to low $\lambda$. The lone black curve is what we would expect to see from a gaussian distribution of components


![cumul_power_pos_eigvec.png](images%2Fcumul_power_pos_eigvec.png)

Let's do the same for negative eigenvalues, red curves correspond to high absolute value of the eigenvalue.

![cumul_power_neg_eigvec.png](images%2Fcumul_power_neg_eigvec.png)

A few points:

- There's definitely a clear pattern where the eigenvectors corresponding to large eigenvalues are much more sharply distributed than those from smaller eigenvalues. 
- A few outlier directions have as much as 80% of their squared sum being concentrated in merely 100 components of the network
- The same basic pattern happens with both negative and positive eigenvalues.
- A plausible hypothesis for what's happening here is that total loss is very sensitive to a minority of the parameters, probably corresponding to the biases of the network and the learned batchnorm variances, and this is showing up in the eigenvectors. 


## 6. Are the Eigenvectors Mostly Constant? <a name="6-eigvec-constant"></a>

Now we turn to the question of figuring out how the eigenvectors of the Hessian change over the course of optimization. Visualizing changes in a 26000 by 26000 matrix is non-trivial, so we need a bit of inventiveness to extract some interesting results here.

We will consider the Eigenvectors of the spectrum at 4 different points in training: iterations 2000, 10000, 20000, and 30000. Given a particular eigenvector at one of these points, we are interested in asking how close it is to the eigenvectors at the previous point in training. And in particular we want to know how close this eigenvector is to the old eigenvectors with roughly the same eigenvalue as it was. 

Meaning, is the new eigenvector just changing in a random direction in parameter space, or is its change biased towards directions that had roughly the same eigenvalues?

To be specific, consider the eigenvector $v_\lambda$ at iteration 10000 with eigenvalue $\lambda$, and project it onto the old eigenspace at iteration 2000 made up of the old vectors $w_{\lambda}$, to get the squared components $a(\lambda, \lambda^\*)$:

$$ a(\lambda, \lambda^\*) = \big(v_\lambda \cdot w_{\lambda^\*}\big)^2$$

In the images below, the x axis corresponds to $\lambda^*$, the old eigenvalue, and the different colored curves correspond to different values of the new $\lambda$, with red curves having high eigenvalues. The y-axis is plotting $a(\lambda, \lambda^\*)$, the squared power of the component at the relevant old eigenvalue.

To take into account the fact that there are many more small eigenvalues than large ones, we plot the power $a(\lambda, \lambda^\*)$ summed in a region around $\lambda^*$ which scales exponentially with $\lambda^\*$. This way we answer the question "how much do eigenvectors with eigenvalues between $10^{-4}$ and $10^{-3}$ contribute to the power of the new eigenvector with value $$10^{-1}"

![sim_2000_10000.png](images%2Fsim_2000_10000.png)

![sim_10000_20000.png](images%2Fsim_10000_20000.png)

![sim_20000_30000.png](images%2Fsim_20000_30000.png)

comments:

- The eigenvectors are in fact changing significantly. No change at all would correspond to each curve being a single infinitely thin spike, since we'd be projecting each eigenvector onto vectors orthogonal to it.
- Vectors with a new value $\lambda$ are *most similar* to the old vector with value $\lambda$, i.e. each curve in the graphs have maximums that correspond to their own $\lambda$ values
- There's a smooth and predictable bias towards the change happening in vectors with neighboring eigenvalues. A vector with a high eigenvalue won't suddenly change in directions with low eigenvalue, it'll only rotate into directions that are in a neighborhood of itself in terms of eigenvalue. This is surprising, as we might've imagined a "spike-and-slab" model, where an eigenvector stays most similar to itself, but apart from that, just rotates in a random direction in parameter space. This is not what seems to happen here.
- The vectors seem to change the most in the early iterations, we see that in the graph from iteration 20000 to 30000, many more of the eigenvectors have sharp power curves, corresponding to the fact that they're mostly staying similar to their old eigenvectors.


## 7. A plausible hypothesis: Narrowing Valleys <a name="7-valley-hypothesis"></a>
While the stochastic quadratic approximation predicts that loss should drop when we drop the learning rate, it *does not* predict that the maximum eigenvalue should rise, nor does it predict that negative eigenvalues remain until the end of training. Can we build an example of a non-quadratic 2D function which exhibits these two properties, or do we need high dimensions to explain these phenomena? The answer turns out to be that **yes**, we can construct such a function. Consider $f(x,y) = \log \bigg( 0.2(x-0.8)^2 + y^2 * \exp(5x+1) + 1 \bigg)$, which we now plot:

![curving_valley_1.png](images%2Fcurving_valley_1.png)

This function has a global minimum at $(0.8, 0)$, with tightening walls in the $y$ direction as we increase the value of $x$. We can also imagine injecting noise by optimising $f(x+\epsilon_x, y+\epsilon_y)$, $\epsilon_x \sim N(0, \sigma_x^2)$, $\epsilon_y \sim N(0, \sigma_y^2)$. Now plotting the normalized gradient:

![curving_valley_gradient.png](images%2Fcurving_valley_gradient.png)

Something interesting is happening with the x component of the gradient, when $y$ is far from 0, the x component of the gradient seems to be pointing *away* from the minimum. Only when $y$ becomes very small does the $x$ derivative begin to point towards the minimum. Let's label in blue the regions of $(x,y)$ where the x-gradient is negative:

![curving_valley_negative_gradx.png](images%2Fcurving_valley_negative_gradx.png)

So the $y$ direction needs to already be mostly optimised before gradient descent can progress in $x$, but the noise in our function means that **there's a lower limit to how small $y$ can be at a given learning rate!**. If $y$ oscillates with a standard deviation of roughly $0.05$, we will never be able to optimise $x$ beyond roughly $0.4$, because the optimisation will spend a lot of time in regions of $(x,y)$ where the x-gradient is negative. Only by lowering the learning rate can we get $y$ closer to $0$ and get the x-gradient to be positive.

What about the negative eigenvalues? Does this simplified model predict that we'll observe negative values? Let's plot the regions of $(x,y)$ where the Hessian has at least one negative eigenvalue (here in red):

![curving_valley_neg_eigenvals.png](images%2Fcurving_valley_neg_eigenvals.png)

Again we see that only a thin band around $y=0$ is without negative eigenvalues. If the noise in $y$ is large enough that gradient descent keeps oscillating outside the thin blue strip, we will keep observing negative eigenvalues in the Hessian. **This explains why the negative eigenvalues don't get optimised away!** The noise in $y$ prevents the optimisation from reaching regions without negative eigenvalues.

This simplified model also provides a possible explanation for the "Edge Of Stability" effect (eigenvalues increase as we decrease learning rate): as we decrease the learning rate, we decrease our variability in $y$, which lets us reach regions of the landscape where the valley walls are much narrower, and hence have much larger eigenvalues.

### The Overall Story According to the Stochastic Narrow Valley Hypothesis

1. We begin optimisation at some random point in the landscape, gradient descent quickly descents down high eigenvalue directions until it gets into equilibrium with the noise in those directions, then we oscillate in the high $\lambda$ directions with some variance $s^2$. However, the low-but-positive- $\lambda$ directions take longer to get optimised, and they benefit from keeping the learning rate higher for longer.
2. The oscillation in the high eigenvalue direction is preventing optimisation from occuring in the narrowing directions, because some significant fraction of oscillations bring the network into regions of the landscape where the gradient is pushing it *away* from the minimum.
3. While we oscillate in the high- $\lambda$ directions, negative eigenvalues don't go away because we keep jumping over the narrow region where the negative values would disappear.
4. When we finally decrease the learning rate (or increase the batch size) by some fixed amount, two things happen: first, the high- $\lambda$ directions drop to a lower equilibrium level, which quickly drops the loss. Then, because we're now oscillating at a lower level, we can correctly "see" the gradient in the narrowing valley directions, this lets us optimise those directions, leading to a further drop in loss. 
5. We settle into a new equilibrium at some point down the narrow valley, the largest eigenvalue of the Hessian increases to reflect the narrowing walls of the new equilibrium point (Edge of Stability effect), and the whole process repeats at the next learning rate drop.

This story seems to concisely explain the "4 mysteries" of training neural networks that we considered at the beginning of this post. Sharp loss decreases, high-batch-low-lr equivalence, the edge of stability, and persistent negative eigenvalues are all effects that naturally fall out of a stochastic landscape where some directions have narrowing valleys.

## 8. Testing the Narrowing Valley Hypothesis <a name="8-testing-valley"></a>

### High Eigenvalues Gate Access to the Correct Low Eigenvalue directions
The narrowing valley hypothesis makes one unambiguous prediction that we should be able to test in realistic networks: **if we optimise the loss function in the subspace defined by the highest eigenvalues of the hessian, the number of negative eigenvalues of the hessian at that local minimum should decrease**. In the context of the toy function from the previous section, that corresponds to finding the thin blue strip in the previous figure. i.e. optimising the y-dimension should bring us within a region of parameter space where there are no more negative eigenvalues.

Note also that this is a non-trivial prediction. If the loss landscape could be well approximated merely by a quadratic function where some of the eigenvalues were negative (i.e. the typical saddle shape), then we would not expect that minimizing the positive eigendirections would have any effect at all on the negative spectrum. Nor would we expect the positive directions to influence the negative ones if the loss could be factorised into $f(x) = f_{+}(v_1, ... v_n) \times f_{-}(w_1, ... , w_n)$, a product of the positive eigenspace and the negative eigenspace.

To test this prediction in our small but non trivial model. We pick again the iteration=10000 checkpointed network, and perform *full batch* SGD for 1000 iterations with lr=1e-3 and momentum=0.9 in the subspace defined by the top n eigenvectors, where n will vary on a logarithmic scale from 6 to 2000. After having found the local minimum within that subspace, we compute the bottom 2000 negative eigenvalues of the network at that point (computing the whole Hessian is too expensive here), and plot the negative spectrum for multiple values of n, the number of top eigenvectors we optimise:

![curving_valley_testing.png](images%2Fcurving_valley_testing.png)

And we see a very robust decrease in the magnitude of negative eigenvalues as we optimise more and more high eigenvalues.

[//]: # (graph with total negative eigenpower vs optimised high eigenvalue dimensions)

## 9. Conclusion and Further Questions <a name="9-conclusion"></a>

Future directions:
- Are there many different narrowing valleys we could fall down into? i.e. if we're oscillating at equilibrium at some learning rate, does the particular point at which we decide to drop the learning rate send us down different narrowing valleys? Or does it not matter?
- How much support do the high eigenvectors have over the data? i.e. are the high eigenvalues due to all datapoints having large dependence on those directions, or do a small number of points have an outsized impact on them?
- Do these results generalise to larger Resnets, what about transformer architectures?
- Are the very small (in absolute terms) eigenvalues important to minimize? Could we restrict the optimisation to a few of the highest eigenvectors as well as the negative vectors, and not lose meaningful performance?
- What, if any, is the connection with the Lottery Ticket Hypothesis?
- Which features of the loss landscape are responsible for overfitting? i.e. should we **want** to go down the narrowing valley, or is going down the valley the price we pay for needing to find the minimum of the high eigenvalues?
- What features of the data and/or the learned representations are responsible for the high/low eigenvalues, and the narrowing effect of the landscape? 
- Can we determine an optimal learning rate schedule from our knowledge of the eigenvalue spectrum and the noise level in each dimension? 
- Can we design architectures whose loss functions exhibit less of a narrowing effect, thereby being trainable with higher learning rates? 
- Can we design architectures where most of the high- $\lambda$ vectors are sequestered to a small number of network components? This would allow us to use much higher learning rates in the rest of the network without risking divergence.
- How do we efficiently minimise noise in the larger eigenvalues while still making large steps in the low-eigenvalue directions?

## Appendix: Derivation of Equilibrium Distribution for SGD with Momentum <a name="sgd-mom-derivation"></a>

When adding momentum, the equations become:

$$ a_{n+1} = \gamma a_n + (2\lambda (x_n + \epsilon))$$

$$   x_{n+1} = x_n - \alpha a_{n+1}$$

$$   a_{n+1} = \sum_{i=0}^n \gamma^{n-i} \bigg(2\lambda (x_i + \epsilon_i) \bigg)$$

$$    x_{n+1} = x_n - \alpha \sum_{i=0}^n \gamma^{n-i} \bigg(2\lambda (x_i + \epsilon_i) \bigg)$$

$$    x_{n+1} = x_n \bigg(1- 2\alpha\lambda\bigg) - \alpha\bigg(2\lambda\epsilon_n + \sum_{i=0}^{n-1} \gamma^{n-i} \bigg(2\lambda (x_i + \epsilon_i) \bigg) \bigg)$$

If we assume that subsequent $x_i$ are roughly independant at equilibrium, we get for the variance $s^2$ of the equilibrium distribution of SGD with momentum:

$$    s^2 = s^2 \bigg(1- 2\alpha\lambda\bigg)^2 + \bigg(2\alpha\lambda \sigma\bigg)^2 + \alpha^2 \lambda^2 (s^2 + \sigma^2)\bigg(\sum_{i=0}^{n-1} (2\gamma^{n-i})^2\bigg)$$

$$   \bigg(\sum_{i=0}^{n-1} (2\gamma^{n-i})^2\bigg) \rightarrow \frac{4\gamma^2}{1-\gamma^2}$$

$$   s^2 = s^2 \bigg(1- 2\alpha\lambda\bigg)^2 + \bigg(2\alpha\lambda \sigma\bigg)^2 + \alpha^2 \lambda^2 (s^2 + \sigma^2)\bigg(\frac{4\gamma^2}{1-\gamma^2}\bigg)$$

$$   s^2\bigg(1 - \bigg(1- 2\alpha\lambda\bigg)^2 - \frac{4\alpha^2\lambda^2\gamma^2}{1-\gamma^2} \bigg) = \bigg(2\alpha\lambda \sigma\bigg)^2 + \frac{4\alpha^2\sigma^2\lambda^2\gamma^2}{1-\gamma^2}$$

$$    s^2\bigg(1 - \alpha\lambda- \frac{\alpha\lambda\gamma^2}{1-\gamma^2} \bigg) = \alpha\lambda\sigma^2 + \frac{\alpha\sigma^2\lambda\gamma^2}{1-\gamma^2}$$

$$    s^2\bigg(1 - \frac{\alpha\lambda}{1-\gamma^2} \bigg) = \frac{\alpha\sigma^2\lambda}{1-\gamma^2}$$

$$    s^2 = \frac{\alpha \sigma^2 \lambda}{1-\gamma^2 - \alpha\lambda} $$




