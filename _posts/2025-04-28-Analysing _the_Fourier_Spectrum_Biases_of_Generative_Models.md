---
layout: distill
title: Analysing The Fourier Spectrum Biases of Generative Models
description: This blog goes in depth in reviewing images in frequency domain and check whether generative models are able to properly reconstruct these images
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# anonymize when submitting 
authors:
  - name: Anonymous 


bibliography: 2025-04-28-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Viewing Images in Frequency Domain
  - name: Analysis of Bias in GANs
  # you can additionally add subentries like so
    subsections:
    - name: Setting Up the Generative CNN Structure
    - name: ReLU as a Fixed Binary Mask
    - name: Onto The Analysis of Filter Spectrum
  - name: Frequency bias in Diffusion Models
  - name: Mitigation of Frequency Bias Using Spectral Diffusion Model
---
## Viewing Images in Frequency Domain

We are used to viewing images in the spatial domain only. But there is another way to view an image, i.e. the frequency domain. We can calculate the frequency content in an image using the 2D discrete Fourier transform. A 2D discrete fourier transform maps a grayscale image $I \in \mathbb{R}^{H \times W}$ to the frequency domain as follows :
$$
\hat{I}[k, l] = \frac{1}{HW} \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} e^{-2\pi i \frac{x \cdot k}{H}} \cdot e^{-2\pi i \frac{y \cdot l}{W}} \cdot I[x, y]
$$

Here, *k*=0,1,2...*H*-1 and *l*=0,1,2,...*W*-1. So it outputs an image in the frequency domain of size ${H \times W}$. Here $\hat{I}[k, l]$ is a complex value at the pixel I[x, y] . For example, the first image is of a cat viewed in the spatial domain and the second image is of the same cat viewed in the frequency domain.

###insert image

Frequency basically refers to the rate of change of colour in an imgae. The smooth regions (where colour or pixel intensities don't change much) in an image correspond to low frequency while the regions containing edges and granular features (where colour changes rapidly) like hair, wrinkles etc correspond to high frequency.

We calculate the power spectral density by squaring the magnitudes of the Fourier components. We can calculate the reduced spectrum S i.e. the azimuthal average over the spectrum in normalized polar coordinates $r \in [0, 1]$, $\theta \in [0, 2\pi)$

$$
\tilde{S}(r) = \frac{1}{2\pi} \int_{0}^{2\pi} S(r, \theta) \, d\theta \quad \text{with} \quad r = \sqrt{\frac{k^2 + l^2}{\frac{1}{4}(H^2 + W^2)}} \quad \text{and} \quad \theta = \text{atan2}(k, l)
$$

Since images are discretized in space, the maximum frequency is determined by the Nyquist frequency. For a square image, $H = W$, it is given by $f_{\text{nyq}} = \sqrt{k^2 + l^2} = \frac{H}{\sqrt{2}}$, i.e. for $r = 1$.

For the above image in frequency domain, the power spectrum is given:



The power spectrum of natural images follows a power law i.e.
$\frac{1}{f^\alpha}$
with $\alpha$ ~ 2
A more complete model of the mean power spectra (using polar coordinates) can be written as

$$
E[|I(f, \theta)|^2] = \frac{A_s(\theta)}{f^{\alpha_s(\theta)}}
$$

in which the shape of the spectra is a function of orientation. The function $A_s(\theta)$ is an amplitude scaling factor for each orientation and $\alpha_s(\theta)$ is the frequency exponent as a function of orientation. Both factors contribute to the shape of the power spectra.
From here we can see that in natural images, the power spectrum is high in the low frequency region and low in the high frequency region. This is intuitive as we expect any natural image to have more smoother regions than edges and complex fine-grained textures.


## Analysis of Bias in GANs

Now, let us analyze the Generative Adversarial Networks(GANs) spectral deficiencies. In this section we show that the ability of GANs to learn a distribution is significantly biased against the high spatial frequencies. Some works have earlier attributed this merely to scarcity of high frequencies in natural images, recent works have shown that this is not so. There are two main hypotheses that have been proposed for the spectral biases; one attributes it to the employment of upsampling operations, and other attributes it to linear dependencies in Conv filter , i.e. , the size of the kernel deployed in the generator network. In addition to this, some works such as [add references]  show that downsampling layers also cause missing frequencies in discriminator. This issue may make the generator lacking the gradient information to model high-frequency content, resulting in a significant spectrum discrepancy between generated images and real images. We take up these hypotheses in the remainder of this section.

### Setting Up the Generative CNN Structure

We’ll start by setting up the structure of a generative CNN model, which typically consists of a series of convolutional layers with filters that learn different features. Our CNN is structured as a stack of convolutional layers, with each layer represented as:

[Insert image for a cnn network]

$$ H_{l+1}^i = \text{Conv}_l^i(H_l) = \sum_c F_{l}^{i,c} * \text{Up}(\sigma(H_l^c)) $$

where:

- $\text{H}_l$ : The feature map at layer *l*.
- $F_{l}^{i,c}$: A convolutional filter at layer *l*, of size ${k_l} \times \text{k}_l$ , that connects input channel *c* to output channel *i*.
- $\text{Up}(\cdot)$ : The upsampling operator, which increases the spatial dimensions, helping generate higher-resolution outputs.
- $\sigma(\cdot)$: A non-linearity function, typically a ReLU. 

**Inputs:**
The initial feature map is represented by $H_1$ with shape $d_0$ $\times$ $d_0$ .

**Parameters:**
The model parameters are $W$ (weights for each layer).

**Layers:**
The network is built from convolutional layers, each generating new feature maps based on its input. Each layer also performs upsampling and non-linear transformations to increase resolution and control spatial frequencies.

Before starting the analysis of filter spectrum, we first need to introduce the idea of viewing ReLU as a fixed binary mask. Why do we need to do this? We'll look at it in just a moment.

### ReLU as a Fixed Binary Mask

Considering ReLUs to be the activation $\sigma(\cdot)$, they then can be viewed as fixed binary masks in the neighbourhood of the parameter $W$. Here, this means that for small variations in the parameters $W$, the activation pattern of the ReLU units (which inputs are passed and which are zeroed) does not change and since the ReLU outputs are determined by the sign of the pre-activation values, these signs only change at specific boundaries in the parameter space, ensuring binary mask remains fixed within any given region. We will now attempt to prove this.

This proof focuses on showing that in a finite ReLU-CNN, the set of parameter configurations where the scalar output of the network crosses zero (i.e., changes sign) has a measure of zero. What is a measure zero? A set of measure zero essentially means that the set occupies "negligible space" in the parameter space. In high-dimensional spaces like $\mathbb{R}^n$, measure-zero sets can often be thought of as lower-dimensional "slices" (e.g., lines, points, or surfaces) within the larger space. While they may exist mathematically, they are effectively insignificant in the context of the full parameter space.

Mathematically, \
We are working with a scalar ouptut $\mathcal{f}(W)$ of a convolutional layer in a finite ReLU-CNN. Therefore, the function depends on the weight parameter *W* and the latent input $H_{1}$. Now, we need to show that for any neighbourhood around $W$, the output of the function is entirely non-negative or entirely non-positive.
This means proving that the set of parameters where 𝑓 changes sign within every neighborhood of 𝑊 (i.e., it crosses zero somewhere in every neighborhood) has measure zero.
$$\implies G = \{ W \in \mathcal{W} \mid \forall N(W), \exists U, V \in N(W) : f(U) < 0 < f(V) \}$$

where $\mathcal{N}(W)$ represents the neighbourhood of $W$. $G$ captures the parameter values $W$ where $f(W)$ crosses zero in every neighborhood. Therefore, our objective becomes to show that $G$ has measure zero.

A finite ReLU-CNN has a finite number of neurons and, hence, a finite number of ReLU activations. Each ReLU activation behaves like a piecewise linear function that "splits" the parameter space into regions. \
$\implies$ For any fixed configuration of active/inactive neurons, $𝑓(𝑊)$ becomes a polynomial function of $𝑊$. Thus, for each configuration of ReLU activations, $𝑓(𝑊)$ behaves as a polynomial, with each configuration yielding a different polynomial form.

A polynomial function on $\mathbb{R}^n \text{ to } \mathbb{R}$ has a measure zero set of zero-crossings in the parameter space (Caron & Traynor, 2005).  Intuitively, this means that the solutions to $f(W)=0$ occupy "negligible" space in the parameter space. \
$\implies$ a finite set of such polynomials also has a measure zero set of zero-crossings. $\therefore$ $G$ is also a measure zero set.

Finally, this reasoning holds for any scalar output $f$ of the network, at any spatial location or layer. Given that there are only a finite number of such outputs in a finite network, the measure of $G$ for all outputs is still zero, thereby completing the proof.

To summarize, the proof hinges on the fact that with ReLU activations, each layer's output depends on whether each neuron is active or inactive. For any fixed set of active/inactive states, the network's output behaves as a polynomial with respect to the parameters. Since polynomials only have zero-crossings on a measure zero subset of the parameter space, the overall network exhibits non-negative or non-positive output behavior *almost* everywhere in the parameter space.

This implies that *almost* all regions of the parameter space are "stable" in terms of sign, and this stability is a result of the ReLU non-linearity creating a finite set of polynomial behaviors for 𝑓.



#### Why This Matters

The key consequences and takeaways of this result are:

**Simplifies Frequency Control:** Since the ReLUs act like fixed binary masks, they don’t introduce additional variability. The network's spectral characteristics become easier to analyze because the ReLUs don’t actively change the frequency content in these neighborhoods.

$\implies$ **Shifts Control to Filters:** The network’s ability to adjust the output spectrum depends more on the convolutional filters ${F}_l^{i,c}$ than on the non-linear ReLUs.

### Onto The Analysis of Filter Spectrum

Now that we have set up the base, we now move onward with analyzing the effect of convolutional filters on the spectrum.

The filters ${F}_l^{i,c}$ in each convolutional layer are now the primary tools for shaping the output spectrum. Thus, the filters try to carve out the desired spectrum out of the input spectrum which is complicated by:

1. Binary masks (ReLU) which altihough dont create new frequencies, but distort what frequencies are passed onto next layer, and aliased by upsampling.
2. Aliasing from Upsampling.

Now, take anytwo spatial frequency components $U =\mathcal{F}_{l}^{i,c} (u_0, v_0)$ and $$V = \mathcal{F}_{l}^{i,c} (u_1, v_1)$$ on the kernel $F_l$ of $l$'th convolution layer spatial dimension $d_l$ and filter size $k_l$, at any point during training.
Let $G_l$ be a filter of dimension $\mathbb{R}^{d_l \times d_l}$. Because of it's dimension, it is and unrestricted filter that can hypothetically model any spectrum in the output space of the layer. Hence, we can write $F_l$ 
as a restriction of $G_l$ using a pulse P of area $k_l^2$ :
$$ F_l = P.G_l \tag{1}$$
$$ P(x,y) = P(x, y) =
\begin{cases} \tag{2}
1, & \text{if } 0 \leq x, y < k_l \\ 
0, & \text{if } k_l \leq x,y \leq d_l  
\end{cases} $$

Applying Convolution Thm on $F_l$:
$$\mathcal{F}_l = \mathcal{F}_{P} \cdot \mathcal{G}_l = \mathcal{F}\{P\} * \mathcal{F}\{G_l\} \tag{3}$$
where $\mathcal{F}(\cdot)$ represents the $d_l$ point DFT.

From (1), the Fourier Transform of $P(x,y)$ is given by:

$$\mathcal{F}\{P(x, y)\}(u, v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} P(x, y) e^{-i 2 \pi (u x + v y)} \, dx \, dy$$

$$\quad \implies \mathcal{F}\{P(x, y)\}(u, v) = \int_0^{k_l} \int_0^{k_l} e^{-i 2 \pi (u x + v y)} \, dx \, dy \quad $$ 

$\text{Evaluating wrt x:}$
$$\begin{equation} \tag{4}
\int_0^{k_l} e^{-i 2 \pi u x} d x=\frac{1-e^{-i 2 \pi u k_l}}{i 2 \pi u}=k_l \operatorname{sinc}\left(\frac{u k_l}{d_l}\right)
\end{equation}$$

$\text{Evaluating wrt y:}$
$$\begin{equation} \tag{5}
\int_0^{k_l} e^{-i 2 \pi v y} d y=\frac{1-e^{-i 2 \pi v k_l}}{i 2 \pi v}=k_l \operatorname{sinc}\left(\frac{v k_l}{d_l}\right)
\end{equation}$$

$\text{Combining these results, the Fourier transform of P(x,y) is:}$
$$\begin{equation} \tag{6}
\mathcal{F}\{P(x, y)\}(u, v)=k_l^2 \operatorname{sinc}\left(\frac{u \kappa_l}{d_l}\right) \operatorname{sinc}\left(\frac{v k_l}{d_l}\right)
\end{equation}$$

When the function is sampled, aliasing causes the spectrum to repeat periodically in the frequency domain. Each repetition of the sinc function at integer multiples of the sampling frequency creates a periodic aliasing pattern. In case of P(x,y) the function transforms into:

$$\begin{equation}\tag{7}
\operatorname{Sinc}(u, v)=\frac{\sin \left(\frac{\pi u k_l}{d_l}\right) \sin \left(\frac{\pi v k_l}{d_l}\right)}{\sin \left(\frac{\pi u}{d_l}\right) \sin \left(\frac{\pi v}{d_l}\right)} e^{-j \pi(u+v)\left(\frac{k_{l}-1}{d_l}\right)}
\end{equation}$$

Here’s a breakdown of the components: \
$\sin(\frac{\pi u k_l}{d_l})$ : This is the sinc function scaled by the ratio of $k_l$ and $d_l$, which determines how the spatial box function in the spatial domain transforms in the frequency domain.

The phase term $e^{-j \pi(u+v)\left(\frac{k_{l}-1}{d_l}\right)}$ : This accounts for a shift in the frequency domain. This phase shift arises due to the position of the box function in the spatial domain. This ensures that the Fourier transform reflects the correct location of the box function.

Calculating for the correlation between $U$ and $V$:

$$\begin{equation}
\operatorname{Cov}[U, V]=\operatorname{Cov}\left[\operatorname{Sinc} * \mathcal{F}\left\{G_l\right\}\left(u_0, v_0\right), \operatorname{Sinc} * \mathcal{F}\left\{G_l\right\}\left(u_1, v_1\right)\right]
\end{equation}$$

To expand this covariance term, we express $U$ and $V$ in terms of the sinc function and the frequency components of $G_l$:

$$
\begin{aligned}
U & =\sum_{u, v} \operatorname{Sinc}(u, v) \cdot \mathcal{F}\left\{G_l\right\}\left(u_0-u, v_0-v\right) \\
V & =\sum_{\hat{u}, \hat{v}} \operatorname{Sinc}(\hat{u}, \hat{v}) \cdot \mathcal{F}\left\{G_l\right\}\left(u_1-\hat{u}, v_1-\hat{v}\right)
\end{aligned}
$$

$$\implies \operatorname{Cov}[U, V]=\operatorname{Cov}\left(\sum_{u, v} \operatorname{Sinc}(u, v) \cdot \mathcal{F}\left\{G_l\right\}\left(u_0-u, v_0-v\right), \sum_{\hat{u}, \hat{v}} \operatorname{Sinc}(\hat{u}, \hat{v}) \cdot \mathcal{F}\left\{G_l\right\}\left(u_1-\hat{u}, v_1-\hat{v}\right)\right)$$

This expands to:
$$
\operatorname{Cov}[U, V]=\sum_{u, v} \sum_{\hat{u}, \hat{v}} \operatorname{Sinc}(u, v) \operatorname{Sinc}^*(\hat{u}, \hat{v}) \cdot \operatorname{Cov}\left(\mathcal{F}\left\{G_l\right\}\left(u_0-u, v_0-v\right), \mathcal{F}\left\{G_l\right\}\left(u_1-\hat{u}, v_1-\hat{v}\right)\right)
$$

Since $G_l$ is assumed to have independent frequency components (with variance $\sigma^2$ for each component), the covariance between any two distinct components is zero, while the variance of each component is $\sigma^2$. Therefore, the covariance term simplifies because we only need to consider the terms where $(u,v) = (\hat{u},\hat{v})$:

$$
\operatorname{Cov}[U, V]=\sum_{u, v} \operatorname{Sinc}(u, v) \operatorname{Sinc}\left(u_0-u_1-u, v_0-v_1-v\right) \cdot \sigma^2
$$

The final covariance expression simplifies further by factoring out $\sigma^2$ and recognizing the sum as a convolution:

$$
\operatorname{Cov}[U, V]=\sigma^2 \sum_{u, v} \operatorname{Sinc}(u, v) \operatorname{Sinc}\left(u_0-u_1-u, v_0-v_1-v\right)
$$  

Using the definition of convolution, we get:
$$
\operatorname{Cov}[U, V]=\sigma^2 \cdot \operatorname{Sinc} * \operatorname{Sinc}\left(u_0-u_1, v_0-v_1\right)
$$

Since the sinc function is defined over the finite output space $d_l \times d_l$, the convolution integrates to $d_l^2$, giving us:
$$
\operatorname{Cov}[U, V]=\sigma^2 d_l^2 \operatorname{Sinc}\left(u_0-u_1, v_0-v_1\right)
$$

Next, we calculate the variance of $U$ (or similarly for $V$, due to symmetry) using the expression for $U$ from earlier. This is computed as:
$$
\operatorname{Var}[U]=\operatorname{Var}\left(\sum_{u, v} \operatorname{Sinc}(u, v) \cdot \mathcal{F}\left\{G_l\right\}\left(u_0-u, v_0-v\right)\right)
$$

Using independence again, the variance simplifies to:
$$
\operatorname{Var}[U]=\sum_{u, v}|\operatorname{Sinc}(u, v)|^2 \cdot \operatorname{Var}\left(\mathcal{F}\left\{G_l\right\}\left(u_0-u, v_0-v\right)\right)
$$

Substituting the variance $\sigma^2$ of each independent component:
$$
\operatorname{Var}[U]=\sigma^2 \sum_{u, v}|\operatorname{Sinc}(u, v)|^2
$$

The sum over $|Sinc(u,v)|^2$ evaluates to $d_l^2k_l^2$, so:
$$
\operatorname{Var}[U]=\sigma^2 d_l^2 k_l^2
$$

Finally, we calculate the complex correlation coefficient between $U$ and $V$, which is defined as:
$$
\operatorname{corr}(U, V)=\frac{\operatorname{Cov}[U, V]}{\sqrt{\operatorname{Var}[U] \operatorname{Var}[V]}}
$$

Substituting,
$$
\operatorname{corr}(U, V)=\frac{\sigma^2 d_l^2 \operatorname{Sinc}\left(u_0-u_1, v_0-v_1\right)}{\sqrt{\sigma^2 d_l^2 k_l^2 \cdot \sigma^2 d_l^2 k_l^2}}
$$
$$\implies
\operatorname{corr}(U, V)=\frac{\operatorname{Sinc}\left(u_0-u_1, v_0-v_1\right)}{k_l^2}
$$

Now, if the $U$ and $V$ frequencies are diagonally adjacent, then the correlation coefficient becomes:
$$
|\operatorname{corr}(U, V)|=\frac{\sin ^2\left(\frac{\pi k_l}{d_l}\right)}{k_l^2 \sin ^2\left(\frac{\pi}{d_l}\right)}
$$

This result indicates that the correlation between two frequency components in the spectrum of $F_l$ is inversely related to the filter size $k_l$. A larger filter (i.e., higher $k_l$) reduces the correlation between frequencies, enhancing the filter's ability to represent diverse frequencies independently. Conversely, a smaller filter (lower $k_l$) increases correlation, meaning that adjustments to one part of the frequency spectrum impact neighboring frequencies, thereby limiting the filter's effective capacity to separate and individually adjust each frequency component.

In each convolutional layer, the maximum spatial frequency that can be achieved is bounded by the Nyquist frequency(you may refer to {add reference}). This means that a convolutional layer can accurately control spatial frequencies within the range $[0, \frac{d_l}{2d}]$ without aliasing. As a result, the high-frequency components are predominantly generated by the outer layers of the CNN, which have larger spatial dimensions $d_l$ With a fixed filter size $k_l$, an increase in $d_l$ leads to higher correlations across the filter’s spectrum, thereby reducing the filter’s effective capacity to fine-tune individual frequencies. Consequently, outer layers, responsible for creating high frequencies, face more restrictions in their spectral capacity compared to inner layers with smaller $d_l$, which have greater flexibility for spectral adjustments.

Moreover, while only the outer layers can produce high frequencies without aliasing, all layers can contribute to the low-frequency spectrum without this restriction. Thus, the spatial extent of the effective filter acting on low frequencies is consistently larger than that acting on high frequencies. Even if larger filter sizes $k_l$ are used in the outer layers to counterbalance the larger $d_l$ , low frequencies continue to benefit from a larger effective filter size compared to high frequencies, which ultimately results in lower correlation at low frequencies.





## Frequency bias in Diffusion Models

It has been well known that like GANs, diffusion models too show frequency bias. Smaller models fail to fit the high frequency spectrum properly. In general, models have a hard time fitting the reduced spectrum graph especially where the magnitude of a particular frequency is low. Diffusion models first fit the high magnitude parts (which correspond to the low frequency region in natural images). After fitting the low frequency region , it then fits the graph in the high frequency region(or high magnitude regions). Large diffusion models have enough parameters and timesteps to fit the high frequency spectrum as well but small models struggle to do so due to lack of enough timesteps. We shall see a simplified yet elaborate version of the math proof as illustrated in Diffusion Probabilistic Model Made Slim paper. We show that by taking the assumption that the denoising network acts as a linear filter, the math works out such that the reduced spectrum is first fitted for the low frequency(or high magnitude) region in the initial timesteps and later fitted for the high frequency (or low magnitude region). Assuming the denoising network as a linear filter, we get it to work as an optimal linear filter or Weiner filter. The function of this Weiner filter is to minimize the mean squared error between the actual noise and the predicted noise by the filter.

Let the input image that we want to reconstruct be $x_0$
and the $\epsilon$ be white noise of variance 1. $x_t$ is the noised sample at time step t. Hence we can write 
$\mathbf{x}_t = \sqrt{\bar{\alpha}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}} \epsilon$

In the denoising process , let $h_t$ be the filter that is learned . So $h_t^* $ is the optimal filter which minimizes the loss function of the diffusion model $L_t = \|\mathbf{h}_t * \mathbf{x}_t - \epsilon\|^2$
.
Here $* $ denotes the standard convolution operation. The optimal filter solution can be found easily in the frequency domain i.e.
$$ \mathcal{H}_t^*(f) = \frac{1}{\bar{\alpha} |X_0(f)|^2 + 1 - \bar{\alpha}} $$.
Here $\mathcal{H}_t^*(f)$ represents the frequency response of the filter. Here a larger value of $$ \mathcal{H}_t^*(f) $$ means that more of the signal at frequency f will be passed. A smaller value of $$\mathcal{H}_t^*(f)$$ means that the signal at frequency f will be attenuated. $|X_0(f)|^2$
is the power spectrum of the original signal i.e. $X_0$, representing the magnitude of a particular frequency f in the spectrum
Let's inspect the formula carefully. During the denoising phase $\bar{\alpha}$ goes from 0 to 1. 

Now if the optimal filter $\mathbf{h}_t$ is learned, then we approximate $\epsilon$ by $\mathbf{h}_t * \mathbf{x}_t$ so our noising step equation becomes
$$
x_t = \sqrt{\bar{\alpha}} \, x_0 + \sqrt{1 - \bar{\alpha}} \, (h_t * x_t)
$$
Taking the DFT on both sides, we get 
$$
X_t = \sqrt{\bar{\alpha}} \, X_0 + \sqrt{1 - \bar{\alpha}} \, (H_t\times X_t)
$$

Rearranging the equation, we get,
$$
\left( \frac{1 - \sqrt{1 - \bar{\alpha}}}{\sqrt{\bar{\alpha}}} H_t \right) X_t = X_0
$$

Let $\left( \frac{1 - \sqrt{1 - \bar{\alpha}}}{\sqrt{\bar{\alpha}}} H_t \right)$ = $G_t$. Here $G_t$ is the frequency response of a filter $g_t$ which is the optimal linear reconstruction filter. Now this optimal filter minimises the equation 
$$
J_t = \left| G_t X_t - X_0 \right|^2
$$
$$
J_t = \left| G_t(\sqrt{\bar{\alpha}} \mathbf{X}_0 + \sqrt{1 - \bar{\alpha}} \epsilon )- X_0 \right|^2
$$


The equation is approximately equal to 
$$
J_t \approx \left| X_0 \right|^2 \left| 1 - \sqrt{\overline{\alpha}} \, G_t \right|^2 + ({1 - \overline{\alpha}}) \, \left| \epsilon \right|^2 \left| G_t \right|^2
$$
as $\epsilon$ and $X_0$ are uncorrelated. Here $\left| X_0 \right|^2$ is the power spectrum of $X_0$ and $\left| \epsilon \right|^2$ is the power spectrum of white noise which is equal to 1.
So to find this optimal reconstruction filter, we differentiate this equation wrt $G_t$ and equate it to 0.
We get,

$$
\frac{\partial J_t}{\partial G_t} = \left| X_0 \right|^2 \left[ 2 \left( 1 - G_t^* \sqrt{\overline{\alpha}} \right) \left( -\sqrt{\overline{\alpha}} \right) \right] + 2 G_t^* (1 - \sqrt{\overline{\alpha}}) = 0
$$

This gives us 
$$
G_t^*  = \frac{\sqrt{\overline{\alpha}}}{\overline{\alpha} + \frac{1 - \overline{\alpha}}{|X_0|^2}}
$$
Here $ G_t^* $ is the conjugate reconstruction filter. As it is real, $G_t^* = G_t$ .
Hence $G_t  = \frac{\sqrt{\overline{\alpha}}}{\overline{\alpha} + \frac{1 - \overline{\alpha}}{|X_0|^2}}$ is the optimal linear reconstruction filter. The predicted $\hat{X}_0$ = $G_t \times X_t$. So predicted power spectrum $|\hat{X}_0|^2 = |G_t|^2 |X_0|^2$

$$
|X_t|^2 \approx  \, \overline{\alpha} |X_0|^2 + (1 - \overline{\alpha}) |\epsilon|^2 = \overline{\alpha} |X_0|^2 + 1 - \overline{\alpha}    
$$
We can approximate it like This as $X_0$ and $\epsilon$ are uncorrelated. Now, let's analyse the expression $|\hat{X_0}|^2$ = $|G_t|^2 |X_t|^2$ = $$ \frac{\overline{\alpha}}{\left( \overline{\alpha} + \frac{1 - \overline{\alpha}}{|X_0|^2} \right)^2} \left( \overline{\alpha} |X_0|^2 + 1 - \overline{\alpha} \right) $$  
Now, during the initial denoising stages, $\bar{\alpha}  \approx 0$. So in the low frequency region, $|X_0|^2$ is very high, so we make the assumption that ${\overline{\alpha}} \, |X_0| \approx 1$. So in the low frequency region, $|\hat{X_0}^2| \approx |X_0|^2$. In the high frequency region, $|X_0|^2$ is low. So, $|\hat{X_0}|^2 \approx 0$. It can be clearly seen that in the inital denoising steps, the high magnitude signal is reconstructed while the low magnitude signal is approximated to zero. 

In the later stages of the denoising process, $\bar{\alpha} \approx 1$, so regardless of the magnitude of $|X_0|^2$, the value $$ |\hat{X_0}|^2 \approx |{X_0}|^2 $$.

So we can clearly see that the diffusion model is succesfully able to learn the low frequency content in its initial denoising steps and eventually, given enough time steps, it learns the entire spectrum. But small diffusion models lack enough time steps and parameters, so only the low frequency spectrum is learnt well by the model and the predicted high frequency content is less than the ground truth

There is another reason which might contribute to this bias. It is because the loss of a DDPM takes an expectation over the dataset.

$$ \mathcal{L}_{\text{DDPM}} = \int p(\mathbf{x}_0) \, \mathbb{E}_{t, \epsilon} \left[ \left\| \epsilon - s(\mathbf{x}_t, t; \theta) \right\|_2^2 \right] d\mathbf{x}_0 $$

Most images have smooth features and there is a small perecntage of samples have high frequency components, hence p($\mathbf{x}_0$) for such samples is low and they are down weighted in the loss function. Due to their low weight, not much importance is given to reconstruction of high frequency components.


## Mitigation of Frequency Bias Using Spectral Diffusion Model

The main problem with diffusion models is that the small vanilla U-Net cannot incorporate the dynamic spectrum into its loss function. So, the authors of this paper introduce a spectrum-aware distillation to enable photo-realistic generation with small models. The U-Net is replaced with a Wavelet Gating module which consists of a WG-Down and WG-Up network. The WG-Down network takes the Discrete Wavelet Transform of the imput image and outputs 4 images of sub-bands. They are respectively the LL, LH, HL, HH sub-bands. In the LL sub-band,a low-pass filter is applied on the rows and columns of the image and thus captures most of the low-frequency content of the image. The LH band is created by passing a low-pass filter on the rows and high-pass filter on the columns and captures the vertical edges of the image. The HL band is created by passing a high-pass filter on the rows and a low-pass filter on the colums of the image thus capturing the horizontal edges of the image. Finally, the HH sub-band is created by passing a high-pass filter on both rows and columns and thus captures diagonal edges . In essence, the LL sub-band captures the low-frequency details i.e. an approximation of the image, while the LH, HL, HH sub-bands capture the high-frequency details of the image. 

The input image of size $H \times W \times C$ is divided into its corresponding 4 sub-bands (each of size $H/2 \times W/2 \times C$). Next, a soft-gating operation is used to weight these 4 sub-bands and the output feature X' is produced as follows:
$$
X'  = \sum_{i \in \{LL, LH, HL, HH\}} g_i \odot X_i
$$
Here, $\odot$ represents element-wise multiplication. The gating mask is learnt using a learnable feed-forward network.
$$
g_{\{LL, LH, HL, HH\}} = \text{Sigmoid}(\text{FFN}(\text{Avgpool}(\mathbf{X})))
$$
In the WG-Up, the input feature is splitted into 4 chunks as the wavelet coefficients. 
Then, WG is carried out to re-weight each sub-band as before:

$$
X' = \text{IDWT}(g_{LL} \odot X_{LL}, g_{LH} \odot X_{LH}, g_{HL} \odot X_{HL}, g_{HH} \odot X_{HH})
$$
Here IDWT is the inverse Discrete wavelet transform
Hence, they provide the model information of the frequency dynamics as well during training as the network decides which components to pay more attention to while learning the gating mask.

Another method that the authors apply is spectrum-aware distillation. So they distill knowledge from a large pre-trained diffusion model into the WG-Unet . They distill both spatial and frequency knowledge into the WG-Unet using a spatial loss and a frequency loss. Let a noised image $x_t$ be passed into the network. The spatial loss is calculated as:
$$
\mathcal{L}_{\text{spatial}} = \sum_{i} \| \mathbf{X}^{(i)}_T - \mathbf{X}^{(i)}_S \|^2_2
$$
where $\mathbf{X}^{(i)}_T$ and $\mathbf{X}^{(i)}_S$ stand for the pair of teacher/student's output features or outputs of the same scale. A single $1 \times 1$ **Conv** layer is used to align the dimensions between a prediction pair.

The frequency loss is then calculated as :
$$
\mathcal{L}_{\text{freq}} = \sum_i \omega_i \left\| \mathcal{X}_T^{(i)} - \mathcal{X}_S^{(i)} \right\|_2^2, \quad \text{where } \omega = \left| \mathcal{X}^{(i)} \right|^{\alpha}
$$
Here, $\mathcal{X}_T^{(i)}$, $\mathcal{X}_S^{(i)}$ and $\mathcal{X}_0^({i})$ represent the 2D DFT of $\mathbf{X}^{(i)}_T$, $\mathbf{X}^{(i)}_S$ and the resized clean image $x_0$. The scaling factor $\alpha$ is -1 We multiply the with $\omega_i$ as it gives more weight to high frequency content ($\mathcal{X}_0^({i})$ is low, hence $\omega_i$ is high) and less weight to low-frequency content.

So the final loss is:
$$
\mathcal{L} = \mathcal{L}_{\text{DDPM}} + \lambda_s \mathcal{L}_{\text{spatial}} + \lambda_f \mathcal{L}_{\text{freq}}
$$
Here $\lambda_s$ = 0.1 and $\lambda_f$ = 0.1 .





------------------------------------------------------------------------
