1. Denoising diffusion probabilistic models. HO, Jonathan, JAIN, Ajay, et ABBEEL, Pieter @hoDenoisingDiffusionProbabilistic2020
2. Diffusion models beat gans on image synthesis @dhariwalDiffusionModelsBeat2021
3. Classifier-free diffusion guidance @hoClassifierFreeDiffusionGuidance2022
4. High-resolution image synthesis with latent diffusion models @rombachHighResolutionImageSynthesis2022
5. (BONUS) Adding conditional control to text-to-image diffusion models @zhangAddingConditionalControl2023
6. (BONUS) Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning @girdharEmuVideoFactorizing2024
7. (BONUS) Regarder MovieGen / SORA.





### Denoising Diffusion Probabilistic models.

### Langevin
Suppose that we have a score-bases model $s_\theta (x) \approx \nabla_x log p (x)$.

We can use a MCMC to sample from $p(x)$ by only knowing $\nabla_x log p(x)$:
$$
x_{i+1} \leftarrow x_i + \epsilon \nabla_x log p (x) + \sqrt{2 \epsilon} z_i
$$

Using $$ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)2}{2\sigma2}\right) $$
So we need an esimation of $\nabla_x logp(x)$.  We use a similar idea (form of algortihm) in diffusion, but not with $\nabla_x logp(x)$. 

### Denoising Diffusion Probabilistic Model

We have $\mathcal{X} = \{x_1,\ldots,x_n\}$ that follows an unknown PDF $p_{\theta^*}$. The goal is to generate new points from this PDF. We will do it using a noising process and reverse process, according to:
$$
\begin{align}
q(x_t \mid x_{t-1}) &= \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I)\\
q(x_{1:T} \mid x_0) &= \prod_{t=1}^T q(x_t \mid x_{t-1})
\end{align}
$$
![[Pasted image 20241110233216.png]]

We can rewrite the value of $x_T$ ($\alpha_t = 1 - \beta_t$ and $\bar{\alpha_t}$ is the product of the $\alpha_i$ for $i \leq t$):
$$
\begin{align}
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t} \sqrt{1-\alpha_{t-1}} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1}
\end{align}
$$
Let $G_1 \sim \mathcal{N}(0,\sigma_1^2 I)$, $G_2 \sim \mathcal{N}(0,\sigma_2^2 I)$ , the sum of them gives $g_2 \sim \mathcal{N}(0,(\sigma_1^2 + \sigma_2^2) I)$.  We apply this to the two terms on the right:
$$
\begin{align}
x_t &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1}) + 1 - \alpha_t} \bar{\epsilon_t} \\ &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\epsilon_t}
\end{align}
$$
According to Lilian Weng blog (maybe also from the article), taking $\beta_1 < \ldots < \beta_T$ is the best method.

We can then ==directly add noise==$x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon$. The next step is to compute the denoising.

Suppose that we have $x_0$ (i.e. we know from which element of $\mathcal{X}$ we sampled to get to $x_T$). It's easy tractable to compute $q(x_{t-1} \mid x_t,x_0)$:
$$
\begin{align}
q(x_{t-1} \mid x_t, x_0) &= q(x_{t} \mid x_{t-1},x_0) \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} \\
&= \ldots
\end{align}
$$
The whole expression isn't really useful (it is in the notebook), but what is cool is the form $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \mu(x_t,x_0); \gamma_t I)$ where the only factor that uses $x_0$ is $\mu$.
Or, using the expression of $x_t$, we know that $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha_t}} \epsilon)$. Hence, we can have an expression $\mu(x_t,t)$.

The idea of the paper is to ==train a neural network to approximate the function $q(x_{t-1} \mid x_t)$== :

$p_\theta (x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta} (x_t,t), \beta_t I)$.
**Question**: They have replaced $\gamma_t$ by just $\beta_t$, why ? **Hypothesis**: It is too precise with $\gamma_t$ and it is not creative enough. But if we are far from the repartition we want we may try this.

Now, to train a network we need a Loss function. They choose the MSE loss because it is the one that empyrically worked better.
**Note**: They use a different $D_{KL}$: $\sum_x log(\frac{P(x)}{Q(x)})$, and not $\sum_x P(x) log(\frac{P(x)}{Q(x)})$.

Once we have $\mu_\theta$ that has been trained, it is supposed to predict $\frac{1}{\sqrt{\bar{\alpha_t}}} (x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta (x_t,t))$. ( It is equation (9) and (10) in @hoDenoisingDiffusionProbabilistic2020 ), just replace $x_t$ by  its expression in the loss computation.

There final loss is simple and given in (14).

## Diffusion Models Beat GANs on Image Synthesis

They use different benchmarks to show that Diffusion beats GANs. The main (the once cited in the first paper by Ho & al) is **F**réchet **I**nception **D**istance. It computes the distance between generated and ground images, the lower the better.

Theyt describe some improvements that have been made on the results of Ho & Al:
- (Improved denoising diffusion probabilistic models, Nichol and Dhariwal) Fixing the variance $\Sigma_\theta (x_t,t)$ to a constant is sub-optimal (that's what did Ho) with few diffusion steps. They parametrized it with a neural network that outputs $v$ and then goes into the function: $\Sigma_\theta (x_t,t) = exp(v log\beta_t + (1-v) log(\tilde{\beta}_t))$. 
- (same ppl) Use an **hybrid objective for training** that allows to learn both $\epsilon_\theta (x_t,t)$ and $\Sigma_\theta (x_t,t)$.
- (https://arxiv.org/pdf/2010.02502) reformulated this as a non-markovian modelisation, called Denoising Diffusion Implicit Models (DDIM).  Here is a list of ideas and why they are useful:
	1. We have access to a trained DDPM. We need it because DDIM is used to forward the sample process.
	2. We search to have $q_\theta (x_{t-1} \mid x_t,x_0)$ with $x_0$ estimated from $\epsilon_\theta(t)$ (but not through the whole process so really blurry and inaccurate). It can be formulated as:
	   $$ q(x_{t-1} \mid x_t, x_0) = N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}(x_0 - \sqrt{1 - \bar{\alpha}_t}) \epsilon_t (x_t))/\sqrt{\bar{\alpha}_t} + \sqrt{1 - \bar{\alpha}_t} \times \epsilon_t (x_t) + \sigma_t \times \epsilon \text{, with } \epsilon \sim N()0,I$$
	3. With $\sigma_t = 0$, we have a determinist function, which means that we have **consistancy**: one $x_T$ will always give the same $x_0$.
	4. Thanks to maths,  we can accelerate by choosing a subset of diffusion steps $[\tau_1, \dots, \tau_S]$ with $S << T$. One can compute $q(x_{\tau_{i-1}} \mid x_{\tau_i}, x_0)$. That decreases a lot the time of sampling.

Jolicoeur-Martineau founds that the UNet Architecture is the one that improves the most the sample quality over the prev architectures. They made an ablation study on different parameters of the UNet architecture:
![[Pasted image 20250101161743.png]]
To increase the performance of models, they use Adaptive Group Normalization (AdaGN, from Nichol & Dhariwal). It incorportates timestep & class embedding into each residual block of the model after a group normalization operation @wuGroupNormalization2018 .


GANs heavily use class labels, a proof of this is by Lucic et  al. which shows that having synthetic labels works way better in a label-limited regime. Hence they want to incorporate classifier guidance to Diffusion Model. The class information were already added in the AdaGN, but they try other approach. 

We can train a classifier $p_\psi (y \mid x_t,t)$ on noisy images $x_t$ and use the gradient $\nabla_{x_t} log(p_\psi(y \mid x_t,t))$ to guide the diffusion sampling process towards an arbitrary class label $y$. It is score-based method, see https://yang-song.net/blog/2021/score/. 

To sample using both $p_\theta$ a DDPM model and $p_\psi$ a classifier, we can sample transition using $p_{\theta, \psi} (x_t \mid x_{t+1}, y) = Z p_\theta (x_t \mid x_{t+1}) p_\psi (y \mid x_t)$ with $Z$ a normalizing constant. Sadly, $Z$ is not tractable, but Sohl-Dickstein did a derivation (available on the artcle, it is (3) to (10)) and shows that $log(p_\theta (x_t \mid x_{t+1}) p_\psi (y \mid x_t)) \approx log p(z) + C_4$ with $z \sim N(\mu + \Sigma g, \Sigma)$ where $\mu,\Sigma$ is mean and covariance of $p_\theta$ and $p(z)$ is a gaussian (see (10) for full expression) . 

The previous derivation works for stochastic methods. But when the model is determinist, we need something else. If we have $\epsilon_\theta (x_t)$ that predicts the noise added to a sample, we can use the score $\nabla_{x_t} log p_\theta (x_t) = - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta (x_t)$.  We can then substitue in the original formulation (the grad of the log of the product of $p_\theta$ and $p_\psi$) to get $\hat{\epsilon}(x_t) := \epsilon\theta (x_t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} log p_\psi (y \mid x_t)$. They give the whole algorithm to do this implementation.

To read: results.

