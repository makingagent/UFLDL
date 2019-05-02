# UFLDL
Notes and Assignments for UFLDL Tutorial - Python3 code

Depends
------------
```bash
pip install numpy
pip install matplotlib
pip install scipy
```

Sparse Autoencoder
------------
* [Notes](http://hertzcat.com/2018/10/20/ufldl-sparse-autoencoder/)
* [Assignment](https://github.com/hertzcat/UFLDL/blob/master/ex1/ex.pdf)
* Data：`IMAGES.mat`

```bash
python train.py
```

Vectorization
------------
* [Notes](http://hertzcat.com/2018/10/26/ufldl-vectorization/)
* [MNIST](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) (download MNIST data)

```bash
python train.py
```

PCA and Whitening
------------
* [Notes](http://hertzcat.com/2018/11/04/ufldl-pca-and-whitening/)
* [Assignment](http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening)
* Data：`IMAGES_RAW.mat`

```bash
python pca_gen.py
```

Softmax Regression
------------
* [Assignment](http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression)
* Data：`train-images-idx3-ubyte`,`train-labels-idx1-ubyte`,`t10k-images.idx3-ubyte`,`t10k-labels.idx1-ubyte`

```bash
python softmaxExercise.py
```

Self-Taught Learning and Unsupervised Feature Learning
------------
* [Assignment](http://ufldl.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning)
* Data：`train-images-idx3-ubyte`,`train-labels-idx1-ubyte`

```bash
python stlExercise.py
```

Building Deep Networks for Classification
------------
* [Assignment](http://ufldl.stanford.edu/wiki/index.php/Exercise:_Implement_deep_networks_for_digit_classification)
* Data：`train-images-idx3-ubyte`,`train-labels-idx1-ubyte`,`t10k-images.idx3-ubyte`,`t10k-labels.idx1-ubyte`

```bash
python stackedAEExercise.py
```

Linear Decoders with Autoencoders
------------
* [Assignment](http://ufldl.stanford.edu/wiki/index.php/Exercise:Learning_color_features_with_Sparse_Autoencoders)
* Data：`stlSampledPatches.mat`

```bash
python linearDecoderExercise.py
```