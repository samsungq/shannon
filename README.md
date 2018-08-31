# shannon
a tool for measuring *information content* in data. 

## setup
you need python 3 and tensorflow. run the following:
```
python mnist.py --output_dir="/outputs" --trials=3 --train_iters=1000
```

## references
see the following papers for more background:
[1] [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/pdf/1606.00709.pdf) by Sebastian Nowozin, Botond Cseke, Ryota Tomioka, June 2016

[2] [Estimating Divergence Functionals and the Likelihood Ratio by Convex Risk Minimization](http://dept.stat.lsa.umich.edu/~xuanlong/Papers/Nguyen-Wainwright-Jordan-10.pdf) by XuanLong Nguyen, Martin J. Wainwright, and Michael Jordan, November 2010