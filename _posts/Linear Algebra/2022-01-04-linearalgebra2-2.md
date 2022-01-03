---
layout: single
title:  "2.2 Matrix Operation (1)"
categories: [Linear Algebra]
tag: [Linear Algebra, Matrix, Matrix operation]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---







이번 포스팅에서는 matrix를 이용한 연산 중, 덧셈, scalar multiple, 곱셈에 대해서 알아보겠습니다.



### 1) Matrix Equality



Matrix를 연산하기 위해서는 먼저 두 matrix가 같다를 정의해야 합니다.



**Definition : Equality of matrices**



다음의 조건을 만족할 때, 두 matrix $A, B$가 같다고 정의합니다.



(1) $A, B$의 matrix size가 같아야 한다.

(2) 각각의 matrix의 같은 위치에 있는 column이 같아야 한다.



$A$의 행과 열의 수와 $B$의 행과 열의 수 같고, 같은 위치에 있는 column, 또는 같은 위치에 있는 entry 값이 같은 경우 $A$와 $B$는 같다고 정의합니다.





### 2) Sum of matrices, Scalar multiple



**Definition : Sum of matrices**



$A, B$가 $m \times n$ matrix인 경우, $A+B$는 $m \times n$ matrix이고, $A+B$의 column은 $A$의 column과 $B$의 column의 합으로 나타내어집니다. 



즉, $A+B$의 $j$번째 column은 $A$의 $j$번 째 column과 $B$의 $j$번 째 column의 합입니다. 

이를 entry로 설명하면, $A+B$의 $(i, j)$ entry는 $A$의 $(i, j)$ entry와 $B$의 $(i, j)$ entry의 합입니다.



따라서, 두 matrix의 합을 정의하기 위해서는 $A, B$의 matrix size(행과 열 개수)가 같아야 정의합니다. 만약 matrix size가 다르면, matrix의 합을 정의하지 않습니다.



*example*


$$
\begin{bmatrix}4&3&2\\1&2&3\end{bmatrix} + \begin{bmatrix}-2&1&4\\-3&-2&3\end{bmatrix} = \begin{bmatrix}2&4&6\\-2&0&6\end{bmatrix}
$$

$$
\begin{bmatrix}4&3&2\\1&2&3\end{bmatrix} + \begin{bmatrix}-2&1\\-3&-2\end{bmatrix} : \ not \ defined
$$




**Definition : Scalar Multiple**



$r$이 scalar값이고, $A$가 matrix인 경우, scalar multiple $rA$은 column이 $A$의 각각의 column에 $r$ scalar배를 한 matrix입니다.

즉, $rA$의 $j$번 째 column은 $A$의 $j$번 째 column에 $r$을 곱한 값입니다.

이를 entry로 설명하면, $rA$의 $(i, j)$ entry는 $A$의 $(i, j)$ entry에 $r$을 곱한 값입니다.



matrix sum과 scalar multiple을 동시에 적용하면 matrix의 일반적인 합, 차, scalar multiple을 계산할 수 있습니다.



*example* 


$$
A=\begin{bmatrix}1&0&3\\2&2&4\end{bmatrix}, B=\begin{bmatrix}-1&5&3\\2&1&-4\end{bmatrix}
$$


일 때, 


$$
A-2B = A+(-2B) = \begin{bmatrix}1&0&3\\2&2&4\end{bmatrix} + \begin{bmatrix}2&-10&-6\\-4&-2&8\end{bmatrix} = \begin{bmatrix}3&-10&-3\\-2&0&12\end{bmatrix}
$$


으로 계산할 수 있습니다.



**Properties of sum and scalar multiples of matrices**



matrix의 합과 scalar multiple에는 다음과 같은 성질을 가지고 있습니다.



$A, B, C$가 같은 size의 matrix이고, $r, s$가 scalar 값일 때



* $A+B = B+A$ : 덧셈에 대한 교환법칙이 성립합니다.
* $(A+B)+C=A+(B+C)$ : 덧셈에 대한 결합법칙이 성립합니다.
* $A+0=A$ : Zero matrix는 덧셈에 대한 항등원입니다.
* $r(A+B)=rA+rB$ : 덧셈에 대해 분배법칙이 성립합니다.
* $(r+s)(A)=rA+sA$ : scalar multiple에 대해 분배법칙이 성립합니다. 
* $r(sA)=(rs)A$ : scalar끼리 곱한 후 matrix에 곱한 것과 scalar multiple을 연속적으로 행한 것의 결과가 같습니다.







### 3) Matrix multiplication



**Definition : Matrix multiplication**



If $A$ is an $m \times n$ matrix, and if $B$ is an $n \times p$ matrix with columns $\boldsymbol{b_1}, \boldsymbol{b_2}, ..., \boldsymbol{b_p}$, then the product $AB$ is $m\times{p}$ matrix whose column is $A\boldsymbol{b_1}, A\boldsymbol{b_2}, ..., A\boldsymbol{b_p}$


$$
AB = \begin{bmatrix}A\boldsymbol{b_1}&A\boldsymbol{b_2}&...&A\boldsymbol{b_p}\end{bmatrix}
$$
 

위와 같이 $AB$의 column은 $A$와 $B$의 column의 곱으로 정의됩니다. 따라서, $AB$가 정의되기 위해서는 $A\boldsymbol{b_j}$가 정의되어야 하기 때문에, A의 column 개수와 $B$의 row 개수가 같아야지 두 matrix의 곱이 정의됩니다. 만약 $A$의 column 개수와 $B$의 row 개수가 다르다면, $AB$는 정의되지 않습니다. 또한  $A\boldsymbol{b_j}$의 성분 개수는 $m$, 즉  $A\boldsymbol{b_j} \in \mathbb{R}^m$ 이므로,  $AB$는 $m \times p$ matrix입니다.



*example*


$$
B=\begin{bmatrix}-1&5&3\\2&1&-4\end{bmatrix},\ C=\begin{bmatrix}1&0\\3&2\\2&-1\end{bmatrix}
$$


에서, $BC$는


$$
B\boldsymbol{c_1} =\begin{bmatrix}-1&5&3\\2&1&-4\end{bmatrix}\begin{bmatrix}1\\3\\2\end{bmatrix}
=1\begin{bmatrix}-1\\2\end{bmatrix}+3\begin{bmatrix}5\\1\end{bmatrix}+2\begin{bmatrix}3\\-4\end{bmatrix} =
\begin{bmatrix}20\\-3\end{bmatrix}
$$



$$
B\boldsymbol{c_2} =\begin{bmatrix}-1&5&3\\2&1&-4\end{bmatrix}\begin{bmatrix}0\\2\\-1\end{bmatrix}
=0\begin{bmatrix}-1\\2\end{bmatrix}+2\begin{bmatrix}5\\1\end{bmatrix}+(-1)\begin{bmatrix}3\\-4\end{bmatrix} =
\begin{bmatrix}7\\6\end{bmatrix}
$$


따라서


$$
BC = \begin{bmatrix}B\boldsymbol{c_1} & B\boldsymbol{c_2}\end{bmatrix} =
\begin{bmatrix}20 & 7 \\ -3 & 6\end{bmatrix}
$$


입니다.



matrix multiplicaion $AB$의 각각의 column을 보면, $A$ **column들의 linear combination**인 것을 알 수 있습니다. 이 때 $AB$**의** $j$**번 째 column은 weight가 **$\boldsymbol{b_j}$**인 A의 column들의 linear combination입니다.** 

즉, $AB$의 column이 $A$의 column들의 linear comination으로 표현되기 때문에, $AB$가 정의되려면 $A$**의 column 개수와 **$B$**의 row 개수가 같아야만 정의됩니다.** 또한 $A\boldsymbol{b_j}$의 성분 개수가 $m$개이고, $j$가 $1$에서 $p$까지 존재하기 때문에 $AB$는 $m \times p$ matrix가 됩니다.





**(1) Row Column rule for computing** $AB$



위와 같이 matrix multiplication을 정의한 것과 결과가 똑같이 나오는 계산 방법이 있습니다. 



If a product $AB$ is defined, then the entry in row $i$ and column $j$ of $AB$ is the sum of the products of corresponding entries from row $i$ of $A$ and column $j$ of $B$



$AB$가 정의되면, $AB$의 $(i, j)$ entry는 $A$의 $i$번 째 row와 $B$의 $j$번 째 column의 같은 위치에 존재하는 성분끼리 곱한 후 모두 더하여 구할 수 있습니다.



$A$의 $i$번 째 row가


$$
\begin{bmatrix}a_{i1}&a_{i2}&...&a_{in} \end{bmatrix}
$$


이고, $B$의 $j$번 째 column이


$$
\begin{bmatrix}b_{1j}\\b_{2j}\\\vdots\\b_{nj}\end{bmatrix}
$$


일 때, $AB$의 $(i, j)$ entry는 


$$
a_{i1}b_{1j}+a_{i2}b_{2j}+\cdots+a_{in}b_{nj} =\Sigma_k^na_{ik}b_{kj} 
$$
입니다.



*example*


$$
B=\begin{bmatrix}-1&5&3\\2&1&-4\end{bmatrix},\ C=\begin{bmatrix}1&0\\3&2\\2&-1\end{bmatrix}
$$


$BC$의 $(1, 1)$ entry는


$$
-1\times1 + 5\times3 + 3\times2 = 20
$$




$BC$의 $(1, 2)$ entry는


$$
-1\times0 +5\times2 + 3\times(-1)=7
$$




$BC$의 $(2, 1)$ entry는


$$
2\times1+1\times3+(-4)\times2=-3
$$


마지막으로 $BC$의 $(2, 2)$ entry는


$$
2\times0 + 1\times2 + (-4)\times(-1)=6
$$


따라서,


$$
BC=\begin{bmatrix}20 & 7 \\ -3 & 6\end{bmatrix}
$$


가 됩니다.





**Properties of multiplication**



$A$가 $m \times n$  matrix이고, $B$와 $C$가 각각의 성질에서 product가 정의가 되도록 조정되는 matrix일 때 다음의 성질을 가집니다.



* $A(BC)=(AB)C$ : matrix multiplication에는 결합법칙이 성립합니다.
* $A(B+C)=AB+AC$ : matrix multiplication에 대해 분배법칙이 성립합니다.
* $(B+C)A=BA+BC$ : matrix multiplication에 대해 분배법칙이 성립합니다. (곱셈 순서 중요!)
* $r(AB)=A(rB)=(rA)B$ for any scalar $r$ : scalar multiple의 경우 어느 순서에 진행하든 matrix multiplication에 영향을 주지 않습니다.
* $I_mA=A=AI_n$ : identity matrix는 matrix multiplication의 항등원입니다.





Matrix multiplication에서 유의깊게 보아야 할 성질은 다음과 같습니다.



* **교환법칙**이 성립하지 않습니다. 



실수 체계에서는 곱셈에 대한 교환법칙이 성립하지만, matrix 곱셈에서는 교환법칙 $AB=BA$가 성립되지 않습니다.



*example*



$A$: $m \times n$ matrix, $B$: $n \times p$ matrix, $m \neq p$



이 경우 $AB$는 정의되지만 $BA$는 정의되지 않습니다. 따라서 교환법칙이 성립되지 않습니다.



* $AB=AC \Rightarrow B=C$ **명제가 성립하지 않습니다.**

  

실수 체계에서는 $A=0$인 경우를 제외하고는 $AB=AC$이면 $B=C$입니다. 하지만 행렬의 곱셈에서는 $B \neq C$임에도 $AB=AC$를 만족하는 경우가 존재합니다.



*example*


$$
A=\begin{bmatrix}1 & 0 \\0 & 0 \end{bmatrix}, \ B=\begin{bmatrix}1 & 2 \\0 & 3 \end{bmatrix}, \ C=\begin{bmatrix}1 & 2 \\0 & 1 \end{bmatrix}
$$


인 경우


$$
AB=\begin{bmatrix}1 & 2 \\0 & 0 \end{bmatrix}, \ AC=\begin{bmatrix}1 & 2 \\0 & 0 \end{bmatrix}
$$


으로 $B\neq C$지만 $AB=AC$입니다.



* $AB=0 \Rightarrow A=0 \ or \ B=0$ **명제가 성립하지 않습니다.**



실수 체계에서는 $AB=0$인 경우 $A$ 또는 $B$가 0입니다. 하지만 행렬에서는 $A\neq 0$, $B\neq0$임에도 $AB=0$인 경우가 존재합니다.



*example*


$$
A=\begin{bmatrix}1 & 0 \\0 & 0 \end{bmatrix}, \ B=\begin{bmatrix}0 & 0 \\1 & 0 \end{bmatrix}
$$


인 경우


$$
AB = \begin{bmatrix}0 & 0 \\0 & 0 \end{bmatrix}
$$


이 됩니다. $A\neq0$, $B\neq0$이지만 $AB=0$입니다.



지금까지 matrix 연산 중 addition, scalar multiplication, multiplication에 대해 알아보았습니다. 다음 포스트에서는 matrix의 연산 중 transpose와 trace에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!