---
layout: single
title:  "5.3 Diagonalization"
categories: [Linear Algebra]
tag: [Linear Algebra, Diagonalization]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





이번 포스트에서는 diagonalization에 대해서 알아보겠습니다.



<br/>



### 1) Diagonalization



<br/>



#### (1) Efficiency of diagonal matrix



<br/>



Diagonal matrix는 계산상의 이점을 가지고 있습니다.



<br/>



*example*


$$
D = \begin{bmatrix}2 & 0 \\ 0 & 3\end{bmatrix}
$$


일 때,


$$
D^2 = \begin{bmatrix}2^2 & 0 \\ 0 & 3^2\end{bmatrix}, \ \ \ D^3 = \begin{bmatrix}2^3 & 0 \\ 0 & 3^3\end{bmatrix}, \ \ \ D^k = \begin{bmatrix}2^k & 0 \\ 0 & 3^k\end{bmatrix}
$$


임을 알 수 있습니다.



<br/>



*example*


$$
D =\begin{bmatrix}2 & 0 & 0\\ 0 & 3 & 0 \\ 0 & 0 & 1\end{bmatrix}
$$


일 때,


$$
D^{-1} =\begin{bmatrix}\frac{1}{2} & 0 & 0\\ 0 & \frac{1}{3} & 0 \\ 0 & 0 & 1\end{bmatrix}
$$


인 것을 알 수 있습니다. 



<br/>



*example*


$$
A = \begin{bmatrix}7 & 2 \\ -4 & 1\end{bmatrix}, \ \ P =  \begin{bmatrix}1 & 1 \\ -1 & -2\end{bmatrix}, \ \ D= \begin{bmatrix}5 & 0 \\ 0 & 3\end{bmatrix}
$$


 $A$와 $D$는 similar합니다. 즉,


$$
A = PDP^{-1}
$$


입니다. 여기서


$$
A^2 = PDP^{-1}PDP^{-1}=PD^2P^{-1}, \\
A^3 = PD^2P^{-1}PDP^{-1}=PD^3P^{-1}, \\
\vdots\\
A^k=PD^kP^{-1} = \begin{bmatrix}1 & 1 \\ -1 & -2\end{bmatrix} \begin{bmatrix}5^k & 0 \\ 0 & 3^k\end{bmatrix}\begin{bmatrix}1 & 1 \\ -1 & -2\end{bmatrix} = \begin{bmatrix} 2\cdot 5^k -3^k & 5^k -3^k \\ 2\cdot 3^k - 2\cdot 5^k & 2\cdot 3^k - 5^k\end{bmatrix}
$$


인 것을 알 수 있습니다.



위 예시들과 같이 diagonal matrix가 가지는 계산상의 이점이 많습니다. 따라서 $A$에 대해 설명할 때 $A$ 대신 $A$와 similar한 diagonal matrix $D$를 이용하여 설명하는 경우가 많습니다. $A$를 $PDP^{-1}$로 바꾸는 과정을 diagonaization이라고 합니다.



<br/>



#### (2) Diagonalization



<br/>



**Definition : Diagonalization**

