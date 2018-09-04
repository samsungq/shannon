# shannon
measure information content in datasets for supervised learning (eg mnist, cifar, agnews). 

since we dont have the entire distribution p(X) we cannot directly estimate mutual information. as an alternative, we estimate mutual information by training a discriminator to distinguish between (i) marginals p(x) and p(y) and (ii) the joint probabilities p(x, y). 

## references

[1] [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/pdf/1606.00709.pdf) by Sebastian Nowozin, Botond Cseke, Ryota Tomioka, June 2016

[2] [Estimating Divergence Functionals and the Likelihood Ratio by Convex Risk Minimization](http://dept.stat.lsa.umich.edu/~xuanlong/Papers/Nguyen-Wainwright-Jordan-10.pdf) by XuanLong Nguyen, Martin J. Wainwright, and Michael Jordan, November 2010

## setup
you need python3 and tensorflow. run the following:
```
python mnist.py --output_dir="./Outputs/MNIST/Default" --trials=3 --train_iters=1000
```

