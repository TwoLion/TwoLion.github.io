---
layout: single
title:  "7.3 Singular Value Decomposition"
categories: [Linear Algebra]
tag: [Linear Algebra, Singular Value Decomposition]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





이번 포스트에서는 Singular Value Decomposition에 대해 다루어보겠습니다.





<br/>

### 1) Singular Value Decomposition



<br/>



이전 포스트에서 Diagonalization에 대해 배웠습니다. 하나의 행렬을 여러 행렬의 곱으로 분해할 수 있는 지의 여부는 행렬의 중요한 성질 중 하나입니다. Diagonalization의 경우, square matrix에 한정해서 가능하였고, 모든 square matrix가 diagonalization이 가능하지는 않았습니다. 

이번 포스트에서 다루는 Singular Value Decomposition(SVD)는 임의의 행렬 모두에 대해서 적용이 가능합니다! 즉 square matrix가 아니더라도 적용이 가능합니다! 어떠한 방법을 이용하여 다루는지 알아보도록 하겠습니다.



<br/>

#### (1) Singular Value

<br/>

**Definition : Singular Value**



Let $A$ be $m\times n$ matrix. The singular values of $A$ are the square root of the eigenvalues of $A^TA$, denoted by $\sigma_1, ..., \sigma_n$, and they are arranged in decreasing order


$$
\sigma_i = \sqrt {\lambda_i}\ , \ \ \ for \ \ i=1,...n
$$
 

where  $\lambda_i$ is the $i$th eigenvalue of $A^TA$.



$A$의 singular value는 $A^TA$의 eigenvalue를 이용하여 구할 수 있습니다. $A^TA$의 eigenvalue에 root를 씌운 값이 $A$의 singular value가 됩니다. 또한 singular value의 경우 일반적으로 크기순으로 순서를 나열합니다. 



해당 정의에서 확인해야 할 점은  다음과 같습니다.

* $A$의 singular value는 중복을 포함하여 $n$개 존재한다.
* $A^TA$의 eigenvalue는 항상 0보다 크거나 같다. 즉 positive semidefinite이다.



이는 

* $A^TA$는 $n\times n$ matrix이므로, 해당 square matrix의 eigenvalue는 중복도(multiplicity)를 고려하여 최대 $n$개 존재하는 것을 알 수 있습니다. 또한 $A^TA$는 symmetric matrix이므로, $n$개의 eigenvalue 모두 실숫값을 가집니다.

* 모든 $\mathbb R^n$에 속하는 $\boldsymbol x$에 대해서 $\boldsymbol x^T A^TA \boldsymbol x = \|A\boldsymbol x\|^2 \geq 0$를 만족하므로, $A^TA$는 positive semidefinite입니다. 즉 $A^TA$의 eigenvalue가 모두 0보다 크거나 같다는 것을 의미하고, 따라서 eigenvalue에 root를 씌운 singular value가 정의될 수 있습니다.



다음 정리는 singular value decomposition의 기본 개념이 되는 정리입니다.



<br/>

**Theorem**



Suppose $\{\boldsymbol v_1, ..., \boldsymbol v_n\}$ is an orthonormal basis of $\mathbb R^n$ consisting of eigenvectors fo $A^TA$, arranged so that the corresponding eigenvalues of $A^TA$ satisfy $\lambda_1\geq \lambda_2 \geq \cdots \geq \lambda_n$, and suppose $A$ has $r$ nonzero singular values. Then $\{A\boldsymbol v_1, ..., A\boldsymbol v_r\}$ is an orthogonal basis for $ColA$, and $rankA = r$



$A^TA$의 eigenvector를 이용하여 $ColA$의 basis를 구할 수 있고, non-zero singular value의 수가 $A$의 rank가 됩니다. 이 때, $C
olA$의 orthogonal basis가 $\{A\boldsymbol v_1, ..., A\boldsymbol v_r\}$인 점은 singular value decomposition 과정에서 중요하게 사용됩니다. (증명은 appendix에 남겨두겠습니다.)





<br/>

#### (2) Singular Value Decomposition

<br/>

그럼 어떠한 방법으로 singular value decomposition을 하게 되는지 확인해봅시다.



<br/>

**Theorem : Singular Value Decomposition**



Let $A$ be $m\times n$ matrix with rank $r$. Then there exists an $m \times n$ matrix $\Sigma$ such that


$$
\Sigma = \begin{bmatrix}D & 0 \\ 0 & 0 \end{bmatrix}
$$


where diagonal entries of $D$ are the first $r$ singular values of $A$, i.e.


$$
D = diag(\sigma_1, ..., \sigma_r), \ \ \sigma_1\geq \sigma_2 \geq \cdots \geq \sigma_r>0
$$


, and there exists $m\times m$ orthogonal matrix $U$ and $n\times n$ orthogonal matrix $V$ such that


$$
A = U\Sigma V^T
$$


$U$ and $V$ are not uniquely determined by $A$, but diagonal entries of $\Sigma$ are necessarily the singular values of $A$.



The columns of $U$ are called left singular vectors of $A$, the columns of $V$ are called right singular vectors of $A$.



임의의 $m\times n$ matrix $A$는 세 개의 matrix $U, \Sigma, V$로 분해가 가능합니다. $\Sigma$는 $m\times n$ matrix인데, $D$와 zero matrix로 partition된 matrix입니다. 이 때, diagonal matrix $D$의 diagonla entry는 $A$의 singular value로 구성됩니다. 따라서 $\Sigma$는 unique하게 결정되구요. 

$U$의 column을 $A$의 left singular vector, $V$의 column을 $A$의 right singular vector라고 합니다. 



실제 임의의 matrix에 singular value decomposition을 적용하기 위해서는 $A$가 $U, \Sigma, V$로 어떻게 분해되는지 알아야 됩니다. 따라서 위의 정리를 증명해보겠습니다.



<br/>

**Proof**



$A$의 rank는 $r$이므로, $A^TA$의 rank는 $r$입니다. 따라서, $A^TA$의 eigenvalue는 중복을 포함하여 총 $n$개가 나오는데, 이 때, $r$의 eigenvalue가 0이 아닌 값을 가지고, $n-r$개의 eigenvalue는 0이 됩니다. 즉 $A^TA$의 eigenvalue는


$$
\lambda_1\geq\lambda_2\geq \cdots \geq\lambda_r \geq 0 = 0 = 0= \cdots = 0
$$


인 것을 알 수 있습니다. $\lambda_1, ..., \lambda_r, 0$에 해당하는 orthonormal한 eigenvector를 각각 $\boldsymbol v_1, ..., \boldsymbol v_n$이라고 하면


$$
\{A\boldsymbol v_1, ..., A\boldsymbol v_r\}
$$


은 $ColA$의 orthogonal basis가 됩니다. 해당 basis에 속한 벡터를 normalize 시킨 


$$
\{\boldsymbol u_1, ..., \boldsymbol u_r\}, \ \ \boldsymbol u_i = \frac{A\boldsymbol v_i}{\|A\boldsymbol v_i\|}, \ \ i=1, ..., r
$$


은 $ColA$의 orthonormal basis가 됩니다. 


$$
\|A\boldsymbol v_i\| = \sqrt{\|A\boldsymbol v_i\|^2} =\sqrt{\boldsymbol v_i^TA^TA\boldsymbol v_i} =\sqrt{\lambda_i \|\boldsymbol v_i\|^2} = \sqrt{\lambda_i} = \sigma_i
$$

을 만족하므로,


$$
\boldsymbol u_i = \frac{1}{\sigma_i}A\boldsymbol v_i, \ \ i=1, ..., r
$$


이 됩니다. 즉


$$
\sigma_i\boldsymbol u_i = A\boldsymbol v_i, \ \ i=1,..,r
$$


를 만족합니다. 위 식을 이용하여 $U, \Sigma, V$를 만들 예정입니다.



먼저 $U$에 대해서 살펴봅시다. 정리에서 $U$는 $m \times m$ orthogonal matrix입니다. $\boldsymbol u_1, ..., \boldsymbol u_r$끼리는 모두 orthonormal하므로, 추가적으로 $m-r$개의 orthonormal한 vector를 찾아주면 됩니다. $\mathbb R^m$에 속하는 vector이기 때문에, $m-r$개의 orthonormal한 vector를 무조건 찾을 수 있습니다. 그리하여 얻은 $\boldsymbol u_1, ..., \boldsymbol u_m$으로 이루어진 집합


$$
\{\boldsymbol u_1, ..., \boldsymbol u_r, \boldsymbol u_{r+1}, ..., \boldsymbol u_m\}
$$


은 $\mathbb R^m$의 orthonormal basis가 됩니다. 



두 번째는 $\Sigma$입니다. $\Sigma$는 $m\times n$ matrix이고


$$
\Sigma = \begin{bmatrix}D & 0 \\ 0 & 0 \end{bmatrix}
$$


다음의 partition된 matrix로 이루어져 있습니다. 이 때 $D = diag(\sigma_1, ..., \sigma_r)$이구요. 



마지막으로 $V$입니다. $V$는 


$$
V = \begin{bmatrix}\boldsymbol v_1 &\boldsymbol v_2 & \cdots & \boldsymbol v_n \end{bmatrix}
$$


으로 만들어집니다. $\boldsymbol v_1, ..., \boldsymbol v_n$은 $A^TA$의 $\lambda_1, ..., \lambda_r, 0$에 해당하는 크기가 1인 eigenvector입니다. $A^TA$는 symmetric이므로, $V$는 orthogonal matrix입니다. 



다음과 같이 $U, \Sigma, V$를 만들면,


$$
U\Sigma = AV
$$


를 만족합니다. 



$U\Sigma$를 살펴보면


$$
U\Sigma = \begin{bmatrix}\sigma_1\boldsymbol u_1 & \sigma_2\boldsymbol u_2 & \cdots & \sigma_r\boldsymbol u_r & 0 & \cdots & 0 \end{bmatrix}
$$


이고, ($\Sigma$의 (1,1), (2, 2), ..., (r, r) entry를 제외하곤 모두 0이기 때문에 $i\geq r+1$부터 $U\Sigma$값은 0입니다.) 



$AV$를 살펴보면


$$
AV = \begin{bmatrix} A\boldsymbol v_1 & A\boldsymbol v_2 & \cdots A\boldsymbol v_r & A\boldsymbol v_{r+1} & \cdots & A\boldsymbol {v}_n \end{bmatrix}
$$


이고,


$$
A\boldsymbol v_r = \begin{cases} A\boldsymbol v_i & i=1,...,r \\ 0 & i=r+1, ..., n\end{cases}
$$


이므로($i>r$인 경우, $\boldsymbol v_i$에 해당하는 eigenvalue가 0이므로, $A^TA\boldsymbol v_i = 0$이고, $\boldsymbol v_i^TA^TA\boldsymbol v_i = \|A\boldsymbol v_i\|^2 = 0$이므로, $A\boldsymbol v_i=0$입니다.),


$$
AV = \begin{bmatrix} A\boldsymbol v_1 & A\boldsymbol v_2 & \cdots A\boldsymbol v_r & 0& \cdots & 0 \end{bmatrix}
$$


가 됩니다. 이 때,


$$
A\boldsymbol v_i = \sigma_i\boldsymbol u_i, \ \ i=1,...,r
$$




이므로, 


$$
AV = U\Sigma
$$


가 성립합니다. 여기에, $V$는 orthogonal matrix이므로


$$
A = U\Sigma V^T
$$


를 만족합니다.



위 정리를 통해서 임의의 $m\times n$ matrix $A$의 singular value decomposition은 다음의 과정을 통해 구할 수 있습니다.



1. $A^TA$의 eigenvalue를 구하고, $A$의 singular value를 구한다. ($\sigma_1\geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$)
2. Singular value를 이용하여 $\Sigma$를 구한다.
3. $A^TA$의 각각 eigenvalue에 대응하는 크기 1의 eigenvector를 구한다.(eigenvalue의 중복도와 같은 수의 linearly independent한 eigenvector를 구합니다. **해당 eigenvector를 column으로 가지는 matrix가 $V$가 됩니다.**)
4. $\boldsymbol v_1, ..., \boldsymbol v_r$를 이용하여 $\boldsymbol u_1, ..., \boldsymbol u_r$을 구한다. ($A\boldsymbol v_i = \sigma_i\boldsymbol u_i,$ 이므로, $\boldsymbol u_i = \frac{1}{\sigma_i}A\boldsymbol v_i$가 됩니다.)
5. 만약 $r=m$이면 $U = \begin{bmatrix} \boldsymbol u_1 & \cdots & \boldsymbol u_r \end{bmatrix}$이 되고, $r<m$이면, 추가적인 orthonormal한 벡터 $\boldsymbol u_{r+1}, ...,\boldsymbol u_{m}$을 구해 $U$를 만든다. ($\mathbb R^m$에 속한 벡터이므로 반드시 존재합니다. **이 때 Gram-Schmidt process를 이용합니다.**)





<br/>

*Example*


$$
A = \begin{bmatrix} 4 & 11 & 14 \\ 8 & 7 & -2 \end{bmatrix}
$$


다음 $A$ matrix를 분해해보겠습니다.

<br/>

* $A$의 singular value 구하기


$$
A^TA = \begin{bmatrix} 80 & 100 & 40 \\ 100 & 170 & 140 \\ 40 & 140 & 200 \end{bmatrix}
$$


를 이용하여 $A^TA$의 eigenvalue를 구하면


$$
\det(A^TA-\lambda I) = 0 \\
\lambda = 360, \ 90, \ 0
$$


인 것을 알 수 있습니다. 따라서 $A$의 singular value는


$$
\sigma_1 = 6\sqrt{10}, \ \sigma_2 = 3\sqrt{10}
$$


가 됩니다.



<br/>



* $\Sigma$ 구하기



$A$의 singular value를 이용하여 $\Sigma$를 구할 수 있습니다.


$$
\Sigma = \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \end{bmatrix} =\begin{bmatrix} 6\sqrt{10} & 0 & 0 \\ 0 & 3\sqrt{10} & 0 \end{bmatrix}
$$


<br/>



* $A^TA$의 eigenvector 구하기



$\lambda_1=360$일 때


$$
A^TA\boldsymbol x =360 x
$$


를 만족하는 크기 1의 eigenvector $\boldsymbol v_1$는


$$
\boldsymbol v_1 = \begin{bmatrix}\frac{1}{3} \\ \frac{2}{3} \\ \frac{2}{3} \end{bmatrix}
$$
가 되고,



$\lambda_2=90, \ \lambda_3=0$일 때 eigenvector $\boldsymbol v_2, \boldsymbol v_3$는


$$
\boldsymbol v_2 = \begin{bmatrix}-\frac{2}{3} \\ -\frac{1}{3} \\ \frac{2}{3} \end{bmatrix}, \
\boldsymbol v_3 = \begin{bmatrix}\frac{2}{3} \\ -\frac{2}{3} \\ \frac{1}{3} \end{bmatrix}
$$


이 됩니다. 이를 이용하여


$$
V = \begin{bmatrix}\boldsymbol v_1 & \boldsymbol v_2 & \boldsymbol v_3\end{bmatrix} =
\begin{bmatrix}\frac{1}{3} & -\frac{2}{3} & \frac{2}{3}\\ \frac{2}{3} &  -\frac{1}{3} & -\frac{2}{3} \\ \frac{2}{3} & \frac{2}{3} &  \frac{1}{3} \end{bmatrix}
$$


$V$를 구할 수 있습니다.



<br/>

* $\boldsymbol u_1, ..., \boldsymbol u_r$ 구하기



$\boldsymbol u_i = \frac{1}{\sigma_i}A\boldsymbol v_i$임을 이용하여 $\boldsymbol u_1, \boldsymbol u_2$를 구합니다.


$$
\boldsymbol u_1 = \begin{bmatrix}\frac{3}{\sqrt{10}} \\ \frac{1}{\sqrt{10}}\end{bmatrix}, \ 
\boldsymbol u_2 = \begin{bmatrix}\frac{1}{\sqrt{10}} \\ -\frac{3}{\sqrt{10}}\end{bmatrix}
$$




<br/>

* $U$ 만들기



지금 $U$는 $2\times 2$ matrix여야 하는데, $\boldsymbol u_1, \boldsymbol u_2$만으로 충분히 $U$를 만들 수 있습니다. (만약 이전 단계에서 구한  $\boldsymbol u$의 개수가 부족하다면, Gram-Schmidt process를 통해 orthonormal한 $\boldsymbol u$벡터를 찾아주면 됩니다.)


$$
U = \begin{bmatrix}\boldsymbol u_1 & \boldsymbol u_2 \end{bmatrix} = \begin{bmatrix} \frac{3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\ \frac{1}{\sqrt{10}} & -\frac{3}{\sqrt{10}}\end{bmatrix}
$$




정리하면


$$
A = U\Sigma V^T \\

U=\begin{bmatrix} \frac{3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\ \frac{1}{\sqrt{10}} & -\frac{3}{\sqrt{10}}\end{bmatrix}, \ \Sigma = \begin{bmatrix} 6\sqrt{10} & 0 & 0 \\ 0 & 3\sqrt{10} & 0 \end{bmatrix}, \ V=\begin{bmatrix}\frac{1}{3} & -\frac{2}{3} & \frac{2}{3}\\ \frac{2}{3} &  -\frac{1}{3} & -\frac{2}{3} \\ \frac{2}{3} & \frac{2}{3} &  \frac{1}{3} \end{bmatrix}
$$


가 됩니다.





<br/>



*Example*


$$
A = \begin{bmatrix} 1 & -1 \\ -2 & 2 \\ 2 & -2 \end{bmatrix}
$$


다음 matrix를 분해해보겠습니다.



<br/>

* $A$의 singular value구하기


$$
A^TA = \begin{bmatrix}9 & -9 \\ -9 & 9\end{bmatrix}
$$


임을 이용하면


$$
\det(A^TA-\lambda I) = (9-\lambda)^2 -81 = 0 \\
\lambda = 18, \  0
$$


이므로 $A$의 singular value는


$$
\sigma_1 = 3\sqrt 2
$$


 입니다.



<br/>

* $\Sigma$ 구하기



따라서, $\Sigma$는 


$$
\Sigma = \begin{bmatrix} \sigma_1 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} =  \begin{bmatrix} 3\sqrt 2 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
$$


입니다.



<br/>



* $A^TA$의 eigenvector 구하기



$\lambda_1 = 18, \lambda_2=0$일 때의 크기 1인 eigenvector는


$$
\boldsymbol v_1 = \begin{bmatrix}\frac{1}{\sqrt 2} \\ -\frac{1}{\sqrt 2} \end{bmatrix}, \ \boldsymbol v_2 = \begin{bmatrix}\frac{1}{\sqrt 2} \\ \frac{1}{\sqrt 2} \end{bmatrix}
$$


입니다. 이를 이용하면


$$
V = \begin{bmatrix}\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \\ -\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \end{bmatrix}
$$


을 구할 수 있습니다.



<br/>



* $\boldsymbol u_1$ 구하기



$\boldsymbol u_1$은


$$
\boldsymbol u_1 = \frac{1}{\sigma_1}A\boldsymbol v_1 = \begin{bmatrix}\frac{1}{3}\\-\frac{2}{3} \\\frac{2}{3} \end{bmatrix}
$$


으로 구할 수 있습니다.



<br/>



* $U$ 구하기



$U$는 $3\times 3$ matrix여야 하는데, 지금 이전 단계에서 $\boldsymbol u_1$밖에 구하지 못했습니다. 이 후 나머지 벡터 $\boldsymbol u_2, \boldsymbol u_3$는 $\boldsymbol u_1, \boldsymbol u_2, \boldsymbol u_3$가 orthogonormal하도록 만들어주면 됩니다.


$$
\boldsymbol u_1 \cdot \boldsymbol w = \frac{1}{3}w_{1} -\frac{2}{3}w_{2} + \frac{2}{3}w_{3} = 0 \\
w_{1}-2w_{2}+2w_{3}= 0 \\
\boldsymbol w = w_2\begin{bmatrix}2 \\ 1 \\ 0\end{bmatrix} + w_2\begin{bmatrix}-2 \\ 0\\1\end{bmatrix}
$$


여기서,


$$
\boldsymbol w \in Span\{\begin{bmatrix}2 \\ 1 \\ 0\end{bmatrix}, \begin{bmatrix}-2 \\ 0\\1\end{bmatrix}\}
$$




해당 조건을 만족하는 $\boldsymbol w$는 $\boldsymbol u_1$과 orthogonal합니다. 여기서, $\boldsymbol u_2, \boldsymbol u_3$가 서로 orthogonal하도록 만들어주면 됩니다.


$$
\boldsymbol w_1 = \begin{bmatrix}2 \\ 1 \\ 0\end{bmatrix}, \ \boldsymbol w_2 = \begin{bmatrix}-2 \\ 0\\1\end{bmatrix} \\
$$


으로 지정하면, $\boldsymbol w_3$는 위의 subspace에 속하면서 $\boldsymbol w_1$과 orthogonal해야 합니다. Gram-Schmidt process를 이용하면


$$
\boldsymbol w_3 = \boldsymbol w_2 - \frac{\boldsymbol w_1 \cdot \boldsymbol w_2}{\boldsymbol w_1 \cdot \boldsymbol w_1 }\boldsymbol w_1 = \boldsymbol w_2 - \frac{-4}{5}\boldsymbol w_1 = \begin{bmatrix}-\frac{2}{5}\\\frac{4}{5}\\1\end{bmatrix}
$$


가 나와


$$
\boldsymbol u_2 = \frac{1}{\|\boldsymbol w_1\|}\boldsymbol w_1 = \begin{bmatrix}\frac{2}{\sqrt 5} \\ \frac{1}{\sqrt 5} \\ 0\end{bmatrix}  \\

\boldsymbol u_3 = \frac{1}{\|\boldsymbol w_3\|}\boldsymbol w_3 = \begin{bmatrix}-\frac{2}{\sqrt {45}} \\ \frac{4}{\sqrt {45}} \\ \frac{5}{\sqrt{45}}\end{bmatrix}  \\
$$


를 구할 수 있습니다. 따라서,


$$
U = \begin{bmatrix}\boldsymbol u_1 & \boldsymbol u_2 & \boldsymbol u_3 \end{bmatrix} = 
\begin{bmatrix}\frac{1}{3} & \frac{2}{\sqrt 5} & -\frac{\sqrt{2}}{\sqrt{45}}\\-\frac{2}{3} & \frac{1}{\sqrt 5} & \frac{\sqrt 4}{\sqrt{45}}  \\ \frac{2}{3} &0 &\frac{5}{\sqrt{45}} \end{bmatrix}
$$


가 되고,


$$
A = U\Sigma V^T \\

V = \begin{bmatrix}\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \\ -\frac{1}{\sqrt 2} & \frac{1}{\sqrt 2} \end{bmatrix}, \ \Sigma = \begin{bmatrix} 3\sqrt 2 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \ U = \begin{bmatrix}\frac{1}{3} & \frac{2}{\sqrt 5} & -\frac{\sqrt{2}}{\sqrt{45}}\\-\frac{2}{3} & \frac{1}{\sqrt 5} & \frac{\sqrt 4}{\sqrt{45}}  \\ \frac{2}{3} &0 &\frac{5}{\sqrt{45}} \end{bmatrix}
$$
 

가 됩니다.





<br/>



지금까지 Singular Value Decomposition에 대해 알아보았습니다. 다음 포스트에서는 Singular Value Decomposition를 이용한 다양한 matrix의 성질에 대해 다루어보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!





<br/>



### Appendix : Proof of Theorem



<br/>

**Theorem**



Suppose $\{\boldsymbol v_1, ..., \boldsymbol v_n\}$ is an orthonormal basis of $\mathbb R^n$ consisting of eigenvectors fo $A^TA$, arranged so that the corresponding eigenvalues of $A^TA$ satisfy $\lambda_1\geq \lambda_2 \geq \cdots \geq \lambda_n$, and suppose $A$ has $r$ nonzero singular values. Then $\{A\boldsymbol v_1, ..., A\boldsymbol v_r\}$ is an orthogonal basis for $ColA$, and $rankA = r$



<br/>



* **Proof**

<br/>


$$
\{\boldsymbol v_1, ..., \boldsymbol v_n\}
$$


은 $A^TA$의 eigenvector이고, 각각의 eigenvectors에 해당하는 eigenvalue가


$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n
$$


입니다. 여기서 $\lambda_1, ..., \lambda_r\geq 0$이고, 이 후의 $\lambda_{r+1}, ..., \lambda_{n}=0$입니다. 먼저


$$
\{A\boldsymbol v_1, ..., A\boldsymbol v_r\}
$$


가 orthogonal한 것을 먼저 보이겠습니다. $i\neq j$에 대해서


$$
A\boldsymbol v_i \cdot A\boldsymbol v_j = \boldsymbol v_jA^TA\boldsymbol v_i = \lambda_i \boldsymbol v_j \cdot \boldsymbol v_i = 0
$$


을 만족합니다. 이는 $A^TA$는 symmetric matrix이므로, $A^TA$의 서로 다른 eigenvalue에 해당하는 eigenvector는 orthogonal하기 때문입니다. (만약 eigenvalue가 같다면, eigenspace의 dimension과 해당 eigenvalue의 중복도가 같으므로, 해당 eigenspace에서 orthogonal한 벡터를 중복도만큼 만들 수 있습니다.)



두 번째로


$$
\{A\boldsymbol v_1, ..., A\boldsymbol v_r\}
$$


가 $ColA$의 basis인 것을 보이겠습니다. 먼저 


$$
\{A\boldsymbol v_1, ..., A\boldsymbol v_r\} \subset ColA
$$


입니다. $A\boldsymbol v_i$는 $A$의 column의 linear combination이기 때문입니다. 현재 linear independence는 만족하였으므로(orthogonal하므로), 위 집합을 이용하여 span하였을 때 column space가 되는지만 확인하면 됩니다. 이를 확인하기 위해


$$
\{\boldsymbol v_1, ..., \boldsymbol v_n\}
$$


를 생각해봅시다. 해당 벡터들은 orthogonal하고, $\mathbb R^n$에 속해있습니다. 따라서 해당 집합은 $\mathbb R^n$의 basis가 됩니다. 한편


$$
\boldsymbol y \in Col A \\
\Rightarrow \boldsymbol y = A\boldsymbol x, \ \ \boldsymbol x \in \mathbb R^n
$$


이 성립됩니다. 여기서, $\boldsymbol x \in \mathbb R^n$이므로


$$
\boldsymbol x = c_1\boldsymbol v_1 +\cdots + c_n \boldsymbol v_n
$$


이 되고


$$
\boldsymbol y = c_1A\boldsymbol v_1 + \cdots + c_nA\boldsymbol v_n
$$


가 됩니다. 즉,


$$
Span\{A\boldsymbol v_1, ..., A\boldsymbol v_n\} = ColA
$$




를 만족합니다. 그런데, $i>r$부터의 $A^TA$의 eigenvalue는 0이므로, $A\boldsymbol v_i = 0$ 또한 0이 됩니다. 즉 위의 식이


$$
Span\{A\boldsymbol v_1, ..., A\boldsymbol v_r, 0\} = ColA
$$


을 만족하게 되죠. 위의 식에서 $0$을 제외하더라도


$$
Span\{A\boldsymbol v_1, ..., A\boldsymbol v_r\} = ColA
$$


가 성립합니다. 따라서


$$
\{A\boldsymbol v_1, ..., A\boldsymbol v_r\} 
$$


는 $ColA$의 basis가 되고, $rankA = r$이 성립합니다.