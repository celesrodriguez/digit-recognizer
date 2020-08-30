# Digit Recognizer.
We implement a digit classifier and we test it on Kaggle Contest. 
We train our classifier using k-fold cross-validation to reduce overfitting. The following methods are implemented:
1) kNN: for each sample we search the k Nearest Neighbours and we guess its label based on proximity. 
2) PCA + kNN: We use the Iterative Power Method for approximating the dominant eigenvalue and we use Deflation techniques. Then, we combine it with kNN. 

The main purpose is to iteratively achieve better results on metrics such as recall and accuracy.  


## Instructions


1. Clone repository.

```
git clone https://github.com/celesrodriguez/digit-recognizer.git
```

2. Dowload `pybind` and `eigen` as submodules

```
git submodule init
git submodule add https://github.com/eigenteam/eigen-git-mirror
git submodule add https://github.com/pybind/pybind11
git mv eigen-git-mirror eigen
# Elegimos versiones de eigen y pybind
cd pybind11/ && git checkout v2.2.4 && cd ..
cd eigen && git checkout 3.3.7 && cd ..
```

3. Install requirements.

```
pip install -r requirements.txt
```

4. Download data from https://www.kaggle.com/c/digit-recognizer/data and unzip it.

```
cd data && gunzip *.gz && cd ..
```

5. Run jupyter.

```
jupyter lab
```

### Data

In `data/` we have training (`data/train.csv`) and test data (`data/test.csv`).

The training data set has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

