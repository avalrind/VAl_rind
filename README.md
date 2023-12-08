
# About VAlrind 

**VAlrind** is a `Python Library` based on `Numpy` for `Training Neural Networks`, and `Training Graph Neural Networks` There are many `SOTA Libraries` such as `Pytorch`/`Tensorflow`/`HuggingFace` and much more for doing so. 

**But VAlrind have a different approach with NNs.** 

# Working 

It `encodes` the values of a `Matrix` of `(m x n)` into a `Row`/`Col` of `len(row) = m`/`len(col) = n`. Then it uses `Bivariate Functions` to calculate the values of `Matrix` at every intersection of `row`/`col`. 

It calculates the `row`/`col` of the `Final Matrix` (after the Matrix Multiplication), directly from the `row`/`col` of the `Constituent Matrices`. 

Doing so `decreases the Time Complexity` of Matrix Multiplication to `O(m + n)`.

<img src = 'https://i.ibb.co/z2MtZgp/MM.png'>

In this Digram we have used `Multiplication` $f(x , y) = x * y$ as the `Bivariate Function`

# Mathametics

For the Multiplication 
    
We denote the `row`/`col` of the matrices as `f_row`/`f_col`/`s_row`/`s_col`/`o_row`/`o_col`

|||
|---|---|
|`f_row`|`Row of First Matrix`
|`f_col`|`Col of First Matrix`
|`s_row`|`Row of Second Matrix`
|`s_col`|`Col of Second Matrix`
|`o_row`|`Row of Output Matrix`
|`o_col`|`Col of Output Matrix`

If we want to calculate what should be at the position `[0 , 0]` in the `Output Matrix`. The traditional way is to 

```
sum(
    (f_col[0] * f_row) * 
    (s_row[0] * s_col)
) 
```
$$O_{0,0} = \sum((f_-col_0 * f_-row) * (s_-row_0 * s_-col))$$

We can break down the formula as 

$$O_{0,0} = \sum(f_-col_0 * f_-row * s_-row_0 * s_-col)$$

As we try to iterate over the whole matrix 

$$O_{0->n,0->m} = \sum(f_-col_{0->n} * f_-row * s_-row_{0->m} * s_-col)$$

As `f_row` and `s_col` act as constants we can take them out of the `summission`

$$O_{0->n,0->m} = f_-row * s_-col\sum(f_-col_{0->n} * s_-row_{0->m})$$

We introduce some denotions here as 

$$f_-row -> o_-row$$

$$s_-col\sum(f_-col_{0->n} * s_-row_{0->m}) -> o_-col$$

Further denoting this formula as 

$$O = o_-row * o_-col$$

And thus making our formula for the output matrix.

****

## For any Bivariate Function

We denote any Bivariate Function to be (π) and for this Bivariate Function 

$$O_{0,0} = \sum((f_-col_0 (π) f_-row) * (s_-row_0 (π) s_-col))$$

$$=>O_{0->n,0->m} = \sum(f_-col_{0->n} (π) f_-row * s_-row_{0->m} (π) s_-col)$$

$$ => O_{0->n,0->m} = f_-row (π) s_-col\sum(f_-col_{0->n} * s_-row_{0->m})$$

We introduce some denotions here as 

$$f_-row -> o_-row$$

$$s_-col\sum(f_-col_{0->n} * s_-row_{0->m}) -> o_-col$$

Further denoting this formula as 

$$O = o_-row (π) o_-col$$

And thus making our formula for the output matrix.

****
**Note :** Sorry for the inconvinence, because of some ambiguity in account with PyPi, we are not able to fully publish the Library, this will be fixed soon

For the Mean-time use the comannd below in your terminal to install the Library 
```
pip install git+https://github.com/avalrind/VAl_rind
```
