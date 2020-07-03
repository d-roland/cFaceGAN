# Conditional Face GAN

## Abstract

We compare conditional generative adversarial networks (cGANs) with different numbers of conditions. We train progressively larger GANs in order to improve stability and variation and utilize the auxiliary classifier approach to conditioning. Qualitatively, our conditional GANs performed well regardless of the number of conditions and generated images accordingly. Quantitatively, our forty conditions GAN performed worse than models with fewer conditions as measured by Sliced Wasserstein Distance and Inception Score.

## 1. Introduction

Conditional generative adversarial networks which allow a trained generator to output samples that comply with certain conditions have been proposed since the advent of generative adversarial networks [5]. We examine the effect of changing the number of conditions on the training of these conditional GANs as well as the accuracy of output samples in terms of following their conditions. The number of conditions can theoretically affect the accuracy of output samples by limiting the number of samples which match a specific combination of conditions during training. We examine how number of samples changes as number of conditions increases and how that affects our GAN training and generator performance. We train our conditional GAN using labeled facial images so that the trained generator is able to produce face images corresponding to difference facial characteristics. The ability to generate images of a face given certain parameters has applications in avatar generation and gaming. For our method, we plan to train a conditional deep convolutional GAN (DCGAN) as implemented in [16]. To improve output image quality we utilize progressively deeper GANs as in [8].

## 2. Related Work
GANs have drawn lots of attention over recent years asthey excelled at (re-)creating realistic images in multiple contexts: face [18] or age [12] synthesis,  realistic photograph [15] [10] or human pose generation [4] for example. Conditional GANs also witnessed great success as their performances improved significantly over the past years. Latest most visible applications included image to image translation with models such as StarGAN [2] or DualGAN [20], but also text to image synthesis [21] with models such as StackGAN [21] or DCGAN [16]. As performance grew, new metrics were needed for both monitoring and benchmarking: the Frechet Inception Distance (FID) [7] and Inception Score [17] are notably used by most recent models. More recently, the FID got fine tuned to produce the Frechet Joint Distance (FJD) [3]. Very recently, we noticed promising results from the MIT versus the computationally intensive drawback of cGANs [11], which prevents them from being deployed on edge devices like mobile phones, tablets or VR headsets with insufficient hardware resources, memory or power. This could increase their usage in mobile gaming context, for instance for avatar generation or profile picture customization.

## 3. Problem Statement
### 3.1 Creating a conditional generator
We are creating a generator neural network that takes input vector c_in of binary conditions and outputs a random face image that matches those binary conditions.  The generator function also takes in latent input z which is usually random and ensures there is variability in the output image for a given c_in. The output image is an RBG image with size 128×128 pixels. 
### 3.2 The training process
The generator is trained as part of training a conditional generative adversarial network. The inputs to the training process are real facial images along with their associated binary labels. We describe the specifics of how the training process works in section 4. The output of the training process are trained generator and discriminator networks. The trained generator is then used to generate images satisfying input conditions.

## 4. Technical Approach
### 4.1. Progressive GAN architecture and training process
We utilize the network architecture described in [8]. This convolutional architecture grows progressively deeper during training. During training, both the image resolution and network depth increases. The generator and discriminator network depths grow together symmetrically. The final generator and discriminator networks are shown in Figures 1 and 2. 
![alt text](https://github.com/d-roland/cFaceGAN/images/blob/master/.ipynb_checkpoints/image_1.png?raw=true)\
The training starts at a level of detail which corresponds to 4×4 images. For the generator, only the firstblock of two convolutional layers are used. Then there is a final Conv1×1layer which outputs a 4×4 image. As the level of detail changes by one, the output image’s height and width grows by a factor of two, and for the generator network, we add an upsample and two convolutional layers prior to the final Conv1×1. For the discriminator network, when training with 4×4 images, we only have the final block shown in Figure 2. We have a Conv 1×1 layer right before the Minibatch Standard Deviation layer. We refer the reader to the original paper [8] for details on the Minibatch Standard Deviation layer. As the level of detail changes by one, the dimensions of the input image to the discriminator doubles, and we add the next block of two Conv 3×3 layers and Downsample layers. During the entire training, the level of detail progressively changes a total of five times where the images increase from 4×4 to 128×128 pixels. A total of over 6 million images are fed through the training process.
### 4.2. Auxiliary classifer approach to conditioning
Our  baseline  is  a  GAN  which  does  not  take  any  conditions for c_in. We increase the number of conditions to compare against our baseline. The approach we  take to condition our GAN is using the auxiliary classifier approach as described in [14]. The generator network takes in a condition variable c_in as mentioned earlier. The difference between the auxiliary classifier approach versus vanilla conditioning is that the condition vector c_in is not an input to the discriminator. Instead, the discriminator is tasked with labeling the input image with a correct c. In addition to the traditional loss functions minimized for the generator and discriminator, the auxiliary classifer conditioning adds a label penalty to both of their respective loss functions. We use the binary cross entropy loss function for each component of c_in where y_i is the true label and ŷ_i is the sigmoid output from the discriminator for label component i.
In this way, the training of the GAN results in a generator that produces images adhering to the input conditions, and the discriminator not only determines if an image is real but also labels an input image with the correct attributes.

### 4.3. Wasserstein loss function 
For the loss functions we use the Wasserstein loss function [1] which allows for greater stability during training by providing a more stable gradient. The generator loss is:
LWGANG=−Eimageout∼pg[D(imageout)]
The discriminator, or critic, loss is:
LWGAND=−Ex∼pd[D(x)]+Eimageout∼pg[D(imageout)]
where p_d is the distribution of real images and p_g is the distribution of generated images. Note the Wasserstein loss function does not include any cross entropy calculations and reflects the Wasserstein distance between the two probability distributions p_d and p_g. It is also required that the discriminator function D is a K-Lipshitz function for some K[1]. To achieve this, we also include the gradient penalty below in the discriminator loss function [6].
penaltygradient=λEˆx∼Pˆx[(‖∇ˆxD(ˆx)‖2−1)2]
P_x is a uniform sampling of the linear interpolation between P_d and P_g. For further details on Wasserstein loss function and gradient penalty we refer the reader to the original cited papers.

## 5. Dataset
We use the CelebA dataset [13] of 200k+ images with already labeled attributes. These binary attributes are shown in Figure 3. The larger the size the more balanced the attribute in the dataset. The original images are RBG images of size 178 pixels by 218 pixels. We crop the images and only use the center 128 by 128 pixels. Each pixel is stored as an unsigned 8-bit integer, and all images are stored as serialized tf.Example objects in TFRecords. We generate 6 TFRecord files for images at each of the following resolutions: 4×4,8×8,16×16,32×32,64×64,128×128 pixels. We start with 128×128 sized images and downsample the resolution by averaging each 4 pixel cluster. These TFRecord files allow us to readily train our GAN at progressively greater resolutions.

## 6. Experiments
We experimented training different network architectures with our CelebA dataset. We tried various fully connected, deep convolutional, and other architectures.   Our early attempts were unsuccessful as we experienced mode collapse or we produced images that didn’t look like faces.
We finally settled on using progressive GANs [8] which offered the best stability in training and variation in results. We were able to train 5 different progressive GANs, each model taking over 22 hours to train. We show the outputs of our GAN with no conditions, as well of some samples from our GAN with 2, 4, and 40 conditions. Figure 4 shows sample generated images from our fully trained progressive GAN without any conditions.
Figure 5 shows sample generated images from our fully trained progressive GAN with two conditions -male, not smiling.
Figure 6 shows sample generated images from our fully trained progressive GAN with two conditions -female, smiling.
Figure 7 shows sample generated images from our fully trained progressive GAN with four conditions -female, smiling, attractive, no makeup.
Figure 8 shows sample generated images from our fully trained  progressive GAN with four conditions -male, not smiling, not attractive, makeup.
Figure 9 shows sample generated images from our fully trained progressive GAN with forty conditions.

We see that our conditional models perform well in following the input conditions. We can see that our forty condition model produces lower quality images and has less variability given the breadth of the constraints.

### 6.1. Metrics
We first discuss the four evaluation metrics used to eval-uate the output of our trained generators-Sliced WassersteinDistance, Inception Score, Frechet Inception Distance, and Multi-Scale Structural Similarity. Then in 7 we will observe these metrics on the GANs we trained with different number of conditions.

### 6.2. Sliced Wasserstein Distances
Wasserstein distance serves as a metric between two probability measures, but it is often intensive to compute. Hence, we utilize Sliced Wasserstein Distance which is an approximation of Wasserstein distance. The Sliced Wasserstein Distance is calculated by using linear projections of the input distributions from many dimensions to one dimension and then computing the Wasserstein distance between the one-dimensional representations [9].
We sample the Sliced Wasserstein Distance at four differentresolutions similar to [8]. The lower the Sliced WassersteinDistance the better quality the generated images.

### 6.3. Inception Score
The Inception Score is an early quantitative metric for evaluating the quality of generative models. It utilizes the Inception v3 model to classify fake generated images, and the output of the classification represents the conditional probability for each image p(y|x). Then the marginal probabilities p(y) are calculated by  averaging the conditional probabilities in a group of images. The conditional probabilities p(y|x) should have low entropy and the marginalprobabilitiesp(y)should have high entropy which reflects high quality images of a class with good variation. These requirements are formulated using the KL divergence [17].
KL(p(y|x)‖p(y)) =p(y|x)∗(log(p(y|x))−log(p(y)))
The exponent of the KL divergence is then applied to obtain the final Inception Score. The Inception Score varies from 1 to 1000 the total number of classes output by the Inception v3 model. The higher the Inception Scores the higher quality the output generated images.
### 6.4. Frechet Inception Distance
The Frechet Inception Distance allows us to compare how fake generated images compare with real images. Calculating the FID involves passing images into the pre-trained Inception v3 model and using the activations from the pool3 layer. This is the same layer of activations used when calculating the Inception Score. We then calculate the Wasserstein-2 distance between these activations [7]. The Frechet Inception Distance is calculated with the formula
d2((μ1,σ1),(μ2,σ2)) =‖μ1−μ2‖22+Tr(σ1+σ2−2(σ1σ2)1/2)
The lower the Frechet Inception Distance the closer thetwo fake and real distributions and the higher the quality ofthe fake generated images.

### 6.5. Multi-Scale Structural Similarity
Multi-Scale Structural Similarity is a variation of structural similarity which compares high level attributes of images to assess their quality. Structural similarity is calculated by comparing the mean, variance, and covariance of two non-negative inputs. The formula for luminance, contrast, and structure as mentioned in [19].\
MS-SSIM is bounded between -1 and 1 with 1 representing the most similar (identical) images. We evaluate our output images by comparing pairs of generated images using MS-SSIM.

## 7. Results
Figure 10 shows the Sliced Wasserstein Distance calcu-lated on our five GANs with varying number of conditions. It shows that having a low number of conditions has performance roughly in line with our network with no conditions. We do see a difference in quality with our forty condition GAN which has a significantly higher SWD at all resolutions. Looking at Inception Score in Figure 11, we see that our zero, one, two, and four condition models perform similarly, and our 40 condition model scores much worse. Figure 12 shows the Frechet Inception Distance across varying number of conditions. It shows relatively similar performance regardless of having more conditions. Finally, we calculate Multi-Scale Structural Similarity between pairs of generated images in each of our models toassess variation in our output images. We see that variationdecreases as training progresses. In our final trained models, our no-condition model has highest variation as measured by MS-SSIM  followed by our 1- and 40-conditionmodels then the 2- and 4-condition models.

## 8. Conclusion
Overall, all of our conditional GANs appear to work well and generate samples that correspond to the input conditions. Additionally, we see that even images which were not common in the training set (eg. males with makeup) were properly generated by our generator. Analyzing the quantitative metrics, we saw that our forty condition model performed worse in terms of Sliced Wasserstein Distance and Inception Score. 
For future work, we can explore how different condition types affect GAN performance. We only used binary conditions in this project. We would like to investigate how numerical and multi-class conditions affect GAN performance.

## References
[1]  Martin   Arjovsky,   Soumith   Chintala,   and   Leon   Bottou. Wasserstein gan, 2017.\
[2]  Y.  Choi,  M.  Choi,  M.  Kim,  J.  Ha,  S.  Kim,  and  J.  Choo. Stargan:  Unified generative adversarial networks for multi-domain  image-to-image  translation. In 2018  IEEE/CVFConference  on  Computer  Vision  and  Pattern  Recognition,pages 8789–8797, 2018.\
[3]  Terrance  DeVries, Adriana  Romero, Luis Pineda, GrahamTaylor, and Michal Drozdzal. On the evaluation of conditional gans, 07 2019.\
[4]  Yixiao Ge, Zhuowan Li, Haiyu Zhao, Guojun Yin, Shuai Yi,Xiaogang Wang, and Hongsheng Li. Fd-gan:  Pose-guided feature distilling gan for robust person re-identification.  InNeurIPS, 2018.\
[5]  Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, BingXu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014.\
[6]  Ishaan Gulrajani, Faruk Ahmed,  Martin Arjovsky,  VincentDumoulin,  and  Aaron  Courville.Improved  training  ofwasserstein gans, 2017.\
[7]  Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.  Gans trained by a two time-scale update rule converge to a local nash equilib-rium, 2017.\
[8]  Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen.Progressive growing of gans for improved quality, stability,and variation, 2017.\
[9]  Soheil  Kolouri,  Kimia  Nadjahi,  Umut  Simsekli,  RolandBadeau, and Gustavo K. Rohde.  Generalized sliced wasser-stein distances, 2019.\
[10]  C. Ledig, L. Theis, F. Husz ́ar, J. Caballero, A. Cunningham,A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, and W.Shi.   Photo-realistic  single  image  super-resolution  using  agenerative adversarial network.   In2017 IEEE Conferenceon Computer Vision and Pattern Recognition (CVPR), pages105–114, 2017.\
[11]  Muyang Li, Ji Lin, Yaoyao Ding, Zhijian Liu, Jun-Yan Zhu,and Song Han.  Gan compression: Efficient architectures forinteractive conditional gans, 03 2020.\
[12]  Yunfan Liu, Qi Li, Zhenjun Sun, and Tieniu Tan. A3gan: Anattribute-aware attentive generative adversarial network forface aging. 11 2019.\
[13]  Ziwei  Liu,  Ping  Luo,  Xiaogang  Wang,  and  Xiaoou  Tang.Deep learning face attributes in the wild.  In Proceedings ofInternational Conference on Computer Vision (ICCV), De-cember 2015.\
[14]  Augustus  Odena,  Christopher  Olah,  and  Jonathon  Shlens.Conditional  image  synthesis  with  auxiliary  classifier  gans,2016.\
[15]  Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-YanZhu. Gaugan: semantic image synthesis with spatially adap-tive normalization. pages 1–1, 07 2019.\
[16]  Alec Radford, Luke Metz, and Soumith Chintala.  Unsuper-vised representation learning with deep convolutional gener-ative adversarial networks, 2015.\
[17]  Tim  Salimans,  Ian  Goodfellow,  Wojciech  Zaremba,  VickiCheung, Alec Radford, and Xi Chen.  Improved techniquesfor training gans, 2016.\
[18]  Yujun Shen, Ping Luo, Junjie Yan, Xiaogang Wang, and Xi-aoou Tang.  Faceid-gan:  Learning a symmetry three-playergan for identity-preserving face synthesis.   pages 821–830,06 2018.\
[19]  Z. Wang, E. P. Simoncelli, and A. C. Bovik. Multiscale struc-tural similarity for image quality assessment.  InThe Thrity-Seventh Asilomar Conference on Signals, Systems Comput-ers, 2003, volume 2, pages 1398–1402 Vol.2, 2003.\
[20]  Zili Yi, Hao Zhang, Ping Tan, and Minglun Gong.  Dualgan:Unsupervised dual learning for image-to-image translation.pages 2868–2876, 10 2017.\
[21]  Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiao-gang Wang, Xiaolei Huang, and Dimitris Metaxas. Stackgan:Text to photo-realistic image synthesis with stacked genera-tive adversarial networks, 2016.
