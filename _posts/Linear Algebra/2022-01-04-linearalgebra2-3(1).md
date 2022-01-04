---
layout: single
title:  "2.3 Inverse of matrix (1)"
categories: [Linear Algebra]
tag: [Linear Algebra, Matrix, Inverse]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---

















이번 포스트에서는 matrix의 inverse에 대해 알아보겠습니다.

<br/>



### 1) Inverse of a matrix

<br/>



#### (1) Invertible matrix

<br/>



**Definition : Invertible matrix, inverse of a matrix** 



An $n \times n$ matrix $A$ is said to be invertible if there is an $n \times n $ matrix $C$ such that


$$
CA = AC = I
$$


where $I=I_n$, $n \times n$ identity matrix



이 때, $C$를 **inverse of** $A$ 라고 합니다. 



여기서 inverse of $A$인 $C$는 $A$마다 unique하게 존재합니다. (Appendix 참고)



따라서, **inverse of** $A$를 


$$
A^{-1}
$$
으로 표시합니다.



즉, matrix $A$가 invertible하다는 것은 $A^{-1}$가 존재해서


$$
AA^{-1}=A^{-1}A=I
$$




를 만족한다는 뜻입니다.



Invertible matrix 정의에서 중요한 점은 3가지 입니다. 



1. Invertible matrix는 **square matrix**에서 정의됩니다. 
2. $A$와 $A^{-1}$끼리의 multiplication은 **교환법칙이 성립**합니다.
3. $AA^{-1}=A^{-1}A=I$ : multiplication 결과가 identity matrix가 됩니다.



<br/>



**Definition : singular, nonsingular matrix**



invertible matrix와 관련된 matrix로 singular, nonsingular matrix가 있습니다.



**Invertible한 matrix를 nonsingular matrix**라고 하고,



**Not Invertible한 matrix를 singular matrix**라 합니다. 



<br/>



*example*


$$
A=\begin{bmatrix}2 & 5 \\ -3 & 7\end{bmatrix}, \ B= \begin{bmatrix}2 & 5 \\ -3 & 7\end{bmatrix}
$$


일 때, 


$$
AB = BA =\begin{bmatrix} 1 & 0 \\ 0 & 1\end{bmatrix} = I
$$


가 됩니다. 따라서 $A, B$는 invertible matrix(nonsingular matrix)이고, $A^{-1}=B$, $B^{-1}=A$입니다.





<br/>



#### (2) Invertible matrix : $2\times2$ case

<br/>



$2\times2$ matrix인 경우, invertible matrix인지 아닌지 구분하는 방법과 inverse를 구하는 공식이 있습니다.


$$
A=\begin{bmatrix}a & b \\ c & d\end{bmatrix}
$$


이고, $ad-bc \neq 0$을 만족하면, 


$$
A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c &a \end{bmatrix}
$$


입니다. 



이는 $AA^{-1}$를 구해보면


$$
AA^{-1}=A^{-1}A = \frac{1}{ac-bc}\begin{bmatrix} d & -b \\ -c &a \end{bmatrix}\begin{bmatrix} a & b \\ c &d \end{bmatrix} = 
\frac{1}{ad-bc}\begin{bmatrix} ad-bc & 0 \\ 0 &ad-bc \end{bmatrix} = I
$$


를 통해 $A^{-1}$가 됨을 알 수 있습니다. 



이 때
$$
ad-bc
$$


의 값에 따라 matrix가 invertible한지 아닌지 결정이 됩니다. ($ad-bc\neq0$이면 invertible, $ad-bc=0$이면 not invertible합니다.) 따라서 위의 식을 **determinant**라고 정의합니다. 

<br/>



*example*


$$
A=\begin{bmatrix} 3 & 4 \\ 5 &6 \end{bmatrix}
$$


인 경우


$$
A^{-1}=\frac{1}{18-20}\begin{bmatrix}6 & -4 \\ -5 & 3\end{bmatrix} = \begin{bmatrix} -3 & 2 \\ \frac{5}{2} &-\frac{3}{2} \end{bmatrix}
$$


가 됩니다.



<br/>



### (3) Properties of Invertible matrix

<br/>



Invertible matrix는 다음의 성질을 가집니다.



* If $A$ is $n \times n $ matrix, then for each $\boldsymbol{b}$ in $\mathbb{R}^n$, the equation $A \boldsymbol{x}=\boldsymbol{b}$ has the uniquae solution $\boldsymbol{x}=A^{-1}\boldsymbol{b}$
* If $A$ is invertible matrix, then $A^{-1}$ is invertible and $(A^{-1})^{-1}=A$
* If $A, B$ are $n \times n$ invertible matrix, then $AB$ is invertible, $(AB)^{-1}=B^{-1}A^{-1}$
* If $A$ is invertible, then $A^T$ is invertible, and $(A^T)^{-1}=(A^{-1})^T$



특히 첫 번째 성질은 $A$가 invertible하면 matrix equation을 $A^{-1}$를 이용하여 바로 solution을 구할 수 있습니다.

또한, 세 번째 성질은  두개 이상의 invertible matrix의 multiplication으로 일반화가 가능합니다. 



$n \times n$ invertible matrix $A, B, C$에 대해서


$$
(ABC)^{-1} = C^{-1}B^{-1}A^{-1}
$$


이 되고, $n \times n $ invertible matrix $A_1, ..., A_m$에 대해서


$$
(A_1A_2 \cdots A_m)^{-1}=A_m^{-1}A_{m-1}^{-1} \cdots A_2^{-1}A_1^{-1}
$$


이 됩니다. 위의 성질을 이용하여 일반적인 square matrix가 invertible한지 하지 않은지 확인할 수 있습니다. 



위의 성질에 대한 증명은 appendix를 참고해주시기 바랍니다. 



지금까지 invertible matrix와 inverse of matrix에 대해 알아보았습니다. 다음 포스트에서는 실제로 matrix의 inverse를 구하는 방법에 대해 알아보겠습니다. 질문이나 오류가 있으면 댓글로 남겨주세요! 감사합니다!

<br/>



### Appendix : Proof of property



<br/>



#### (1) Uniqueness of the inverse

<br/>



$A$가 invertible하면 $A$의 inverse는 unique합니다.  



* **proof**



$A$가 invertible하므로, $A$의 inverse를 $B$, $C$라고 가정하면


$$
AB=BA=I, AC=CA=I
$$


를 만족합니다.




$$
B = BI = B(AC) = (BA)C = IC = C
$$


가 되어 


$$
B=C
$$


를 만족합니다.

<br/>



#### (2) Properties of invertible matrix

<br/>





* If $A$ is $n \times n $ matrix, then for each $\boldsymbol{b}$ in $\mathbb{R}^n$, the equation $A \boldsymbol{x}=\boldsymbol{b}$ has the uniquae solution $\boldsymbol{x}=A^{-1}\boldsymbol{b}$



* **proof**


$$
A\boldsymbol{x}=\boldsymbol{b}
$$


에서, $A$가 invertible하므로 양변에 $A^{-1}$를 곱해주면


$$
A^{-1}A\boldsymbol{x}=A^{-1}\boldsymbol{b}
$$


이 됩니다. 이 때 좌변에서 $A^{-1}A=I$이므로


$$
\boldsymbol{x}=A^{-1}\boldsymbol{b}
$$


가 됩니다. 따라서 $\mathbb{R}^n$에 속하는 임의의 $\boldsymbol{b}$에 대해서 matrix equation $A\boldsymbol{x}=\boldsymbol{b}$는 unique solution을 가집니다.





<br/>



* If $A$ is invertible matrix, then $A^{-1}$ is invertible and $(A^{-1})^{-1}=A$



* **proof**


$$
A^{-1}A=AA^{-1}=I
$$


입니다. 따라서 


$$
(A^{-1})^{-1}=A
$$


가 됩니다. 



* If $A, B$ are $n \times n$ invertible matrix, then $AB$ is invertible, $(AB)^{-1}=B^{-1}A^{-1}$



* **proof**



$A, B$가 size가 같고 invertible하므로


$$
AB(B^{-1}A^{-1})=AIA^{-1}=AA^{-1}=I \\
(B^{-1}A^{-1})AB=B^{-1}IB=B^{-1}B=I
$$


를 만족하기 때문에, 


$$
(AB)^{-1}=B^{-1}A^{-1}
$$


입니다. 



위 증명은 두 개 이상의  invertible matrices 곱의 inverse를 구할 때 또한 사용할 수 있습니다.



<br/>



* If $A$ is invertible, then $A^T$ is invertible, and $(A^T)^{-1}=(A^{-1})^T$



* **proof**



$A$가 invertible하므로


$$
A^{-1}A=AA^{-1}=I
$$


입니다. 양변에 transpose를 취하면


$$
(A^{-1}A)^T=(AA^{-1})^T=I^T=I
$$


Identity matrix는 diagonal matrix이므로 symmetric합니다. 따라서 위의 식을 정리하면


$$
A^T(A^{-1})^T = (A^{-1})^TA^T=I
$$
가 되어


$$
(A^T)^{-1}=(A^{-1})^T
$$


가 됩니다.