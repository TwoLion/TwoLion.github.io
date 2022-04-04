---
layout: single
title:  "3.2 Properties of Determinant"
categories: [Linear Algebra]
tag: [Linear Algebra, Determinant]
toc: true
author_profile: true#프로필 생략 여부
use_math: true

---









이번 포스트에서는 determinant에 관한 여러 정리와 성질에 대해서 알아보겠습니다.



<br/>

### 1) Triangular matrix and row operation

<br/>



#### (1) Determinant of triangular matrix

<br/>

Triangular matrix의 경우 determinant가 간단하게 구해집니다.



<br/>

**Theorem**



If $A$ is a triangular matrix, then $detA$ is the product of the entries on the main diagonal of $A$





이전 포스트에서, co-factor expansion을 통해 determinant를 계산하는 경우, 0가 많은 column이나 row를 선택하면 determinant 계산이 편리하다는 것을 알 수 있었습니다. 같은 맥락으로, triangular matrix의 경우, diagonal entry를 제외하고 모든 값이 0인 column(upper triangular matrix) 또는 row(lower triangular matrix)가 존재하기 때문에, 이 column과 row를 기준으로 co-factor expansion을 적용하면 결과는 diagonal entry의 곱으로 나온다는 것을 알 수 있습니다.



<br/>



* **sketch of the proof**



$A$가 upper triangular matrix라고 가정해봅시다.


$$
A = \begin{bmatrix}a_{11} & * & * & \cdots & * \\ 0 & a_{22} & * & \cdots & * \\
\vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & \cdots &a_{nn}\end{bmatrix}
$$


$A$의 determinant를 cofactor expansion을 통해  구하기 위해서, 첫 번째 column을 기준으로 co-factor expansion을 적용하면


$$
detA = a_{11}C_{11} = a_{11}(-1)^{1+1}\begin{vmatrix}a_{22} & * & \cdots & * \\ 0 & a_{33} & * & \cdots  \\
\vdots & \vdots & \ddots & \vdots  \\ 0 & 0 &  \cdots &a_{nn}\end{vmatrix}
$$


가 됩니다. $A$에서 첫 번째 column과 row를 제거한 새로운 matrix 역시 triangular matrix입니다. 새로운 matrix의 determinant 역시 첫 번째 column을 이용하여 구하면


$$
detA=a_{11}C_{11}=a_{11}(-1)^{1+1}a_{22}(-1)^{1+1}\begin{vmatrix}a_{33} & * & \cdots & * \\ 0 & a_{44} & * & \cdots  \\
\vdots & \vdots & \ddots & \vdots  \\ 0 & 0 &  \cdots &a_{nn}\end{vmatrix}
$$


가 됩니다. 새로운 matrix 역시 triangular matrix이므로, 위의 방식과 같은 방법으로 적용을 해주면


$$
detA = a_{11}a_{22}\cdots a_{nn}
$$


인 것을 알 수 있습니다. 



정리하면, triangular matrix의 경우 determinant는 diagonal entries의 곱입니다. 



<br/>

#### (2) Effect of elementary row operation

<br/>

Row operation과 determinant와도 특별한 관계가 있습니다.



<br/>

* **Theorem**



Let $A$ be $n \times n$ matrix



1. If a multiple of one row of $A$ is added to another row to produce a matrix $B$ (**replacement**), then $detA=detB$
2. If two rows of $A$ are interchanged to produce $B$ (**interchange**), then $detB=-detA$
3. If one row of $A$ is multiplied by $k$ to produce $B$ (**scaling**), then $detB = kdetA$



If $E$ is $n \times n $ elementary matrix, then


$$
det(E) = \begin{cases} 1 \ \ \ (replacement) \\ -1 \ \ \ (interchange) \\ k \ \ \ (scaling)\end{cases}
$$


즉 row operation을 진행한 matrix와 기존의 matrix의 determinant가 operation 종류에 따라 변화합니다. replacement의 경우 determinant가 그대로 유지되고, interchange의 경우 부호 변화가, scaling의 경우 scaling할 때 곱해준 상수배만큼 determinant에 변화가 생깁니다. 



이전 포스트에서, row operation과 같은 역할을 하는 matrix인 elementary matrix가 존재함을 알고 있다면, elementary matrix의 determinant 또한 위의 정리를 적용하면 쉽게 구할 수 있습니다.



<br/>

#### (3) Unifying (1) and (2)

<br/>

(1) 정리와 (2) 정리를 종합하면, determinant를 구할 수 있는 또다른 방법을 알 수 있습니다.

임의의 $n \times n$ matrix $A$에 대해서 row operation을 통해 **triangular matrix** $B$를 만든 다음, $B$의 determinant와 **row operation에서 발생한 determinant 변화**를 곱해서 $A$의 determinant를 구할 수 있습니다. 



*example*


$$
A = \begin{bmatrix} 1 & -4 & 2 \\ -2 & 8 & -9 \\ -1 & 7 & 0 \end{bmatrix}
$$


$A$의 determinant를 구해보겠습니다.



$A$를 row operation을 통해 triangular matrix로 만들어줍니다.


$$
A =\begin{bmatrix} 1 & -4 & 2 \\ -2 & 8 & -9 \\ -1 & 7 & 0 \end{bmatrix} \sim \begin{bmatrix} 1 & -4 & 2 \\ 0 & 0 & -5 \\ -1 & 7 & 0 \end{bmatrix} \sim \begin{bmatrix} 1 & -4 & 2 \\ 0 & 0 & -5 \\ 0 & 3 & 2  \end{bmatrix} \sim \begin{bmatrix} 1 & -4 & 2 \\ 0 & 3 & 2 \\ 0 & 0 & -5 \end{bmatrix} = B
$$


$A$를 $B$ matrix로 만들어줄 때 사용한 row operation은 replacement(1, 2 번째 연산)와 interchange(세 번째 연산)입니다. 

따라서 
$$
detA = -detB
$$


인 것을 알 수 있습니다. 

$B$는 triangular matrix이므로, 


$$
detB = 1 \times 3 \times -5 = -15
$$


이고,


$$
detA=15
$$


가 됩니다.





<br/>

### 2) Theorems of Determinant and Invertible matrix

<br/>



* **Theorem**



A square matrix $A$ is invertible if and only if $detA\neq0$





Determinant를 통해 어떤 matrix가 invertible matrix인지 아닌지 바로 확인할 수 있습니다. 



<br/>

* **Theorem**



If $A$ has two identical rows of columns, then $detA=0$



동일한 column이나 row를 가지고 있는 경우, determinant는 0입니다. 



<br/>

* **Theorem**



1. $det(kA)=kdetA$
2. $det(AB)=detAdetB$
3. $detA = detA^T$
4. If $A$ is invertible, then $detA^{-1} = \frac{1}{detA}$





각 정리와 성질에 대한 증명은 appendix를 참고하시길 바랍니다.



<br/>

### 3) Invertible Matrix Theorem

<br/>

chapter 2에서 배웠던 invertible matrix theorem에 determinant를 이용하여 새롭게 추가된 명제가 있습니다.



Let $A$ be a square $n \times n$ matrix. Then the following statements are equivalent. That is, for given $A$, the statements are either all true or all false



a. $A$ is an invertible matrix

b. $A$ is row equivalent to the $n \times n $ identity matrix

c. $A$ has $n$ pivot positions

d. The equation $A\boldsymbol{x}=\boldsymbol{0}$ has only the trivial solution

e. The columns of $A$ form a linearly independent set

f. The columns of $A$ span $\mathbb{R}^n$

g. There is an $n \times n $ matrix $C$ such taht $CA=I$

h. There is an $n \times n$ matrix $D$ such that $AD=I$

i. The equation $A\boldsymbol{x}=\boldsymbol{0}$ has at least one solution for each $\boldsymbol{b}$ in $\mathbb{R}^n$

j. $A^T$ is an invertible matrix

**k. $detA\neq 0$ **



<br/>



지금까지 determinant에 대한 여러 성질과 정리에 대해 알아보았습니다. 다음 포스트에서는 determinant를 활용한 Cramer's rule에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글로 남겨주세요! 감사합니다!

<br/>



### Appendix: Proof of the theorem

<br/>



#### 2) Theorems of Determinant and Invertible matrix

<br/>



**Theorem**



A square matrix $A$ is invertible if and only if $detA\neq0$



* **Proof**



$\Rightarrow$



$A$가 invertible하므로, $A$는 $I$, identity matrix와 row equivalent합니다. 즉,


$$
A \sim I
$$


를 만족합니다. $A$에서 $I$로 만드는 row operation을 진행 할 때, determinant는 0이 아닌 실수배의 변화만 존재합니다. 즉,


$$
detA = c \ detI 
$$
여기서 $c$는 상수입니다. 그리고, $detI=1$이므로,


$$
detA \neq 0
$$


이 됩니다. 



$\Leftarrow$



$detA\neq0$ 이므로, $A$의 reduced echelon form이 $I$인 것을 알 수 있습니다. 만약 $A$의 reduced echelon form이 $I$가 아니라면, reduced echelon form의 determinant는 0이 되기 때문에, $detA=0$이 됩니다. 

결국 $A$와 $I$는 row equivalent하므로, $A$는 invertible matrix입니다.



<br/>



**Theorem**



If $A$ has two identical rows or columns, then $detA=0$



* **Proof**



$A$가 identical한 두 개의 row나 column을 가지고 있다면, $A$는 $n$개의 pivot position을 가질 수 없게 되므로, $A$는 invertible하지 않습니다. 따라서 $detA=0$을 만족합니다.





<br/>



**Theorem**





1. $det(kA)=kdetA$
2. $det(AB)=detAdetB$
3. $detA = detA^T$
4. If $A$ is invertible, then $detA^{-1} = \frac{1}{detA}$



* **Proof**



proof of 1. 



$kA$는 $A$ matrix의 모든 row에 $k$배 scaling을 한 matrix로 생각을 하면


$$
det(kA) = k^ndetA
$$
임을 알 수 있습니다.

<br/>



proof of 2.



만약 $A$ 또는 $B$가 invertible하지 않다면, $AB$ 또한 invertible하지 않습니다. 따라서


$$
detAB=detAdetB=0
$$


을 만족합니다.



만약 $A$, $B$ 모두 invertible하면, 두 matrix 각각 elementary matrix의 곱으로 나타낼 수 있습니다. 



$A$를


$$
A = E_kE_{k-1}\cdots E_1
$$


이라고 하면, $AB$는 $B$에 elementary matrix를 곱한 matrix, 즉, $B$에 특정한 row operation을 취한 matrix로 생각할 수 있습니다. 따라서


$$
detAB = det(E_{k}E_{k-1}\cdots E_1B)= det(E_k)det(E_{k-1})\cdots det (E_1)detB = detAdetB
$$


임을 알 수 있습니다.

<br/>



proof of 3.



$detA$를 구할 때 co-factor expansion을 이용하여 구할 수 있습니다. co-factor expansion을 이용하여 구할 때, 특정한 row나 column을 선택하여 determinant를 구합니다. 그런데, $A^T$는 $A$의 column과 row의 위치만 바꾸는 연산이므로, co-factor expansion에 영향을 끼치지 않습니다. 

(예를 들어, $detA$를 구할 때 특정한 column을 선택하여 co-factor expansion을 적용했다면, $detA^T$를 구할 때는 $detA$를 구할 때 선택한 column에 해당하는 row를 선택하여 co-factor expansion을 적용하면 됩니다.)

<br/>



proof of 4.



$AA^{-1}=A^{-1}A=I$이므로


$$
det(AA^{-1})=detAdetA^{-1}=detI=1
$$


따라서


$$
detA^{-1} = \frac{1}{detA}
$$


가 됩니다.
