---
layout: single
title:  "2.3 Inverse of Matrix (3)"
categories: [Linear Algebra]
tag: [Linear Algebra, Matrix, Inverse, IMT]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---







이번 포스트에서는 invertible matrix가 가지는 성질을 정리한 invertible matrix theorem에 대해 알아보겠습니다. 



<br/>

### 1) Invertible Matrix Theorem

<br/>



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



위의 10가지 명제는 동치입니다. a.번을 보면 $A$가 invertible matrix이므로, invertible matrix $A$는 다음 9가지의 성질을 가집니다. invertible matrix theorem은 앞으로 새로운 개념을 배울 때마다 추가될 예정입니다.



<br/>

* **Proof**





<br/>

(1)  a $\Rightarrow$ g $\Rightarrow$ d $\Rightarrow$ c $\Rightarrow$ b $\Rightarrow$ a





<br/>

* a $\Rightarrow$ g 



invertible matrix의 정의에 의해서 $CA=I$를 만족하는 $C$가 존재합니다. ($C=A^{-1}$가 됩니다.)





<br/>

* g $\Rightarrow$ d



$A\boldsymbol{x}=\boldsymbol{0}$에서 양변의 왼쪽에 $C$를 곱해주면


$$
CA\boldsymbol{x}=I\boldsymbol{x}=\boldsymbol{x}=C\boldsymbol{0}=\boldsymbol{0}
$$


가 되어 trivial solution만을 가집니다.





<br/>

* d $\Rightarrow$ c



$A\boldsymbol{x}=\boldsymbol{0}$가 trivial solution을 가진다는 것은, 이 matrix equation과 동일한 solution을 가지는 linear system의 augmented matrix


$$
\begin{bmatrix}A & \boldsymbol{0} \end{bmatrix}
$$


를 풀었을 때, free variable이 존재하지 않습니다. 따라서 $A$의 모든 column이 pivot column이 되고, pivot position 개수는 $n$개입니다.





<br/>

* c $\Rightarrow$ b



$n \times n$ matrix $A$의 pivot column 개수가 $n$개이므로, $A$의 reduced echelon form의 leading entry 개수가 $n$개입니다. 따라서 


$$
A \sim I
$$


를 만족합니다. 





<br/>

* b $\Rightarrow$ a



[저번 포스트](https://twolion.github.io/linear%20algebra/linearalgebra2-3(2)/)의 Theorem에 의해 성립합니다.







<br/>

(2) h $\Rightarrow$ i $\Rightarrow$ c  $\Rightarrow$ b $\Rightarrow$ a $\Rightarrow$ h





<br/>

* h $\Rightarrow$ i



$AD=I$이므로, $A\boldsymbol{x}=\boldsymbol{b}$는 다음과 같이 표현할 수 있습니다.


$$
A\boldsymbol{x}=I\boldsymbol{b} = AD\boldsymbol{b}
$$


따라서 $\boldsymbol{x}=D\boldsymbol{b}$은 위 equation의 solution이 됩니다. 이는 모든 $\boldsymbol{b}\in\mathbb{R}^n$에 대해 성립합니다. 





<br/>

* i $\Rightarrow$ c



$A\boldsymbol{x}=\boldsymbol{b}$가 모든  $\boldsymbol{b}\in\mathbb{R}^n$에 대해 solution이 존재하려면, 다음의 matrix equation과 동일한 solution을 가지는 linear system의 augmented matrix


$$
\begin{bmatrix}A & \boldsymbol{b} \end{bmatrix}
$$


가 모든  $\boldsymbol{b}\in\mathbb{R}^n$에 대해서 solution이 존재해야 합니다. 즉, 임의의 $\boldsymbol{b}$에 대해서 solution이 존재해야 하기 때문에, $A$의 reduced echelon form에 zero row가 존재하면 안됩니다. 따라서 $A$는 $n$개의 pivot position을 가집니다.





<br/>

* c $\Rightarrow$ b $\Rightarrow$ a



(1)에서 증명





<br/>

* a $\Rightarrow$ h

Invertible matrix 정의에 의해 $AD=I$를 만족하는 $D$가 존재합니다. ($D=A^{-1}$)







<br/>

(3) d $\Leftrightarrow$ e



Linear independence 정의에 의해 d와 e는 동치입니다.(d가 e의 정의입니다.)





<br/>

(4) i $\Leftrightarrow$ f



$A\boldsymbol{x}=\boldsymbol{b}$ 에서 $A\boldsymbol{x}$는


$$
A\boldsymbol{x} = x_1\boldsymbol{a_1}+...+x_n\boldsymbol{a_n}
$$


$A$의 column들의 linear combination으로 표현됩니다. 



 모든  $\boldsymbol{b} \in\mathbb{R}^n$에 대해서 $A\boldsymbol{x}=\boldsymbol{b}$ 의 solution이 존재한다는 것은 모든 $\boldsymbol{b}$가 $A$의 columns의 linear combination으로 표현되는 것을 뜻합니다. 즉


$$
\boldsymbol{b} \in Span\{\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}\}
$$


을 만족합니다. 따라서


$$
\mathbb{R}^n \subset Span\{\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}\}
$$


을 만족합니다. 또한, $\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}$은 모두 $\mathbb{R}^n$의 원소이므로, 이들의 linear combination 역시 $\mathbb{R}^n$의 원소입니다. 


$$
\mathbb{R}^n \supset Span\{\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}\}
$$


이므로


$$
\mathbb{R}^n = Span\{\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}\}
$$


을 만족합니다.







<br/>

(5) a $\Leftrightarrow$ j



inverse의 성질 중, $A$가 invertible하면 $(A^T)^{-1}=(A^{-1})^T$임을 만족합니다. 즉 $A^T$ 또한 invertible합니다.







<br/>

지금까지 Invertible Matrix Theorem에 대해서 알아보았습니다. 다음 포스트에서는 Partitioned matrix에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!