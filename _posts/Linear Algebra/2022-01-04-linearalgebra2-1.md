---
layout: single
title:  "2.1 Matrix"
categories: [Linear Algebra]
tag: [Linear Algebra, Matrix]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"

---











두 번째 chapter의 첫 번째 포스트에서는 matrix의 정의와 기본적인 matrix에 대해 알아보겠습니다.



<br/>



### 1) Matrix

<br/>



**Definition : Matrix**



matrix(행렬)는 직사각형 모양의 숫자 배열로, 가로열을 나타내는 row(행)과 세로열을 나타내는 column(열), 각각의 행과 열에 위치하는 성분인 entry로 구성됩니다.



$m \times n $ matrix $A$ 은 row의 개수가 $m$개, column의 개수가 $n$개인 matrix를 뜻합니다. 



$A$ matrix의 $i$번 째 row, $j$번 째 column에 위치하는 scalar 값을 $(i, j)$ entry of $A$이라고 합니다.


$$
A = \begin{bmatrix} a_{11} & a_{12} & ... &a_{1n}\\
a_{21} & a_{22} & ... & a_{2n} \\
\vdots & \vdots & \vdots & \vdots \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}
$$


위 matrix에서 row의 수는 $m$, column의 수는 $n$개이므로, $m \times n$ matrix $A$입니다.



여기서, $A$의 각각의 column은 $m$개의 실수로 구성되어 있으며, $\mathbb{R}^m$에 속하는 vector로 생각할 수 있습니다.



따라서, $A$의 각각의 column을 순서대로 $\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}$이라고 하면 $A$를 다음과 같이 나타낼 수 있습니다.


$$
A = \begin{bmatrix}\boldsymbol{a_1} & \boldsymbol{a_2} & ... & \boldsymbol{a_n}\end{bmatrix}
$$


이 경우 $(i, j)$ entry인 $a_{ij}$는 $\boldsymbol{a_j}$의 $i$번 째 성분으로 해석할 수 있습니다.



$A$의 $(i, j)$ entry 중 $i=j$인 entry를 **diagonal entry**라고 하고, diagonal entry list를 **main diagonal**이라고 합니다.



<br/>



#### Basic Matrix



(1) Zero matrix



모든 entry 값이 0인 matrix를 뜻합니다.

<br/>



*example*


$$
\begin{bmatrix}0 & 0 \\0 & 0\end{bmatrix}
$$




(2) Square matrix



row의 수와 column의 수가 같은 matrix를 뜻합니다. ($n \times n$ matrix)

<br/>



*example* 


$$
\begin{bmatrix}1 & 2 \\3 & 4\end{bmatrix}
$$




(3) Identity matrix



Square matrix에서 main diagonal 값이 모두 1이고, 이를 제외한 모든 entry 값이 0인 matrix를 뜻합니다.

<br/>



*example*


$$
\begin{bmatrix}1 & 0 \\0 & 1\end{bmatrix}, \begin{bmatrix}1 & 0 & 0 \\0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix}
$$




(4) Triangular matrix



Upper triangular matrix : Square matrix에서 main diagonal 아래에 위치한 entry 값이 0인 matrix를 뜻합니다.

Lower triangular matrix : Square matrix에서 main diagonal 위에 위치한 entry 값이 0인 matrix를 뜻합니다.

<br/>



*example*


$$
\begin{bmatrix}1 & 3 & 0 \\0 & 2 & -1 \\ 0 & 0 & 1\end{bmatrix} : upper \ triangular \ matrix
$$



$$
\begin{bmatrix}2 & 0 & 0 \\0 & 4 & 0 \\ -2 & 3 & 1\end{bmatrix} : lower \ triangular \ matrix
$$




Upper triangular matrix의 조건은 main diagonal 아래에 위치한 entry 값이 0이어야 합니다. 따라서 main diagonal과 main diagonal 위에 위치한 entry 값은 어떤 값을 가지든 상관이 없습니다. 이는 lower triangular matrix에도 똑같이 적용됩니다.



(5) Diagonal matrix



Square matrix에서 main diagonal의 값을 제외한 나머지 entry 값이 0인 matrix를 뜻합니다.

<br/>



*example*


$$
\begin{bmatrix}2 & 0 & 0 \\0 & 4 & 0 \\ 0 & 0 & 1\end{bmatrix}
$$


Diagonal matrix의 경우 lower triangular matrix와 upper triangular matrix의 조건을 동시에 만족합니다. 따라서 lower triangular matrix이면서 upper triangular matrix입니다.



지금까지  matrix의 정의, 기본적인 matrix에 대해 알아보았습니다.. 다음 포스트에서는 행렬의 기본 연산에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글로 남겨주세요! 감사합니다!