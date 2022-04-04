---
layout: single
title:  "1.5 Linear Independence"
categories: [Linear Algebra]
tag: [Linear Algebra]
toc: true
author_profile: true #프로필 생략 여부
use_math: true

---













이번 포스트에서는 선형 대수학에서 매우 중요한 개념 중 하나인 linear independence에 대해 알아보겠습니다.



<br/>



### 1) Homogeneous Linear system

<br/>



Linear independence는 homogeneous linear system의 개념을 이용하여 정의합니다.

<br/>



**Definition : Homogeneous Linear system**



Linear system이 다음과 같은 form을 가질 때


$$
A\boldsymbol{x} = \boldsymbol{0}
$$


linear system이 homogeneous하다라고 합니다. ($A:m \times n$ matrix, $\boldsymbol{0}$ : zero vector in $\mathbb{R}^m$)



즉, 일반적인 linear system


$$
A\boldsymbol{x}=\boldsymbol{b}
$$


에서 $\boldsymbol{b}=\boldsymbol{0}$인 경우를 homogenous linear system이라고 합니다.



이 linear system은 일반적인 linear system과 달리 특징이 있습니다. 이는 바로, $A$**에 상관없이 반드시 위 linear system은 solution을 가집니다.** 그 solution은 $\boldsymbol{x}=\boldsymbol{0}$이구요.



$\boldsymbol{x}=\boldsymbol{0}$는 $A$에 상관없이 반드시 성립하기 때문에, homogeneous linear system에서의 solution $\boldsymbol{x}=\boldsymbol{0}$을 **trivial solution**이라고 합니다.

또한, $A$에 따라서 위 homogeneous linear system은 $\boldsymbol{x}=\boldsymbol{0}$이 아닌 다른 solution을 가질 수 있는데, 다른 solution을 **nontrivial solution**이라고 합니다.



linear system에서의 solution 타입은 3가지로, solution이 없는 경우, solution이 하나만 있는 경우, solution이 무수히 많은 경우입니다. homogeneous linear system에서는 solution이 반드시 하나 존재하기 때문에, 우리의 관심은 solution이 하나인지, solution이 무수히 많은지 확인하는 것입니다.



또한, 


$$
A\boldsymbol{x}=\boldsymbol{b}
$$


에서 $\boldsymbol{b} \neq \boldsymbol{0}$인 linear system을 **nonhomogeneous linear system**이라고 합니다.



위의 homogeneous linear system 개념을 이용하여 linear independence를 정의합니다.



<br/>



### 2) Linear Independence

<br/>



Linear independence는 다음과 같이 정의됩니다.

<br/>



**Definition : Linear independence**



An Indexed set of vectors $\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ in $\mathbb{R}^n$ is said to be linearly independent if the vector equation


$$
x_1\boldsymbol{v_1}+x_2\boldsymbol{v_2}+ \cdots + x_p\boldsymbol{v_p}=0
$$


has only trivial solution

The set  $\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ is said to be linearly dependent if there exist weights $c_1, c_2, ..., c_p$, not all zero, such that


$$
c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+ \cdots + c_p\boldsymbol{v_p}=0
$$


The indexed set is linear dependent if and only if it is not linearly independent.



즉 vector가 원소인 집합  $\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ 이 linearly independent하다는 뜻은 이 vector를 이용한 homogeneous linear system이 trivial solution만을 가진다는 뜻입니다. 



만약 homogeneous linear system이 trivial solution이 아닌 non trivial solution을 가진다면, 이 집합은 linearly dependent하다고 합니다. 



여기서, homogeneous linear system의 solution 타입은 solution이 하나만 있거나(trivial solution), solution이 무수히 많은 경우(nontrivial solution) 두 가지밖에 없기 때문에, linearly independent와 linearly dependent는 상반된 개념입니다. 즉, linearly independent하지 않으면 linearly dependent하고, linearly dependent하지 않으면, linearly independent합니다.

<br/>



*example 1)*


$$
\boldsymbol{v_1} = \begin{bmatrix}1 \\ 2 \\ 3\end{bmatrix}, 
\boldsymbol{v_2} = \begin{bmatrix}4 \\ 5 \\ 6\end{bmatrix}, 
\boldsymbol{v_3} = \begin{bmatrix}2 \\ 1 \\ 0\end{bmatrix}
$$


위 세 벡터로 이루어진 집합 $\{\boldsymbol{v_1}, \boldsymbol{v_2}, \boldsymbol{v_3}\}$이 linearly dependent한지 linearly independent한지 확인해보기 위해서 vector equation


$$
x_1\boldsymbol{v_1}+x_2\boldsymbol{v_2}+x_3\boldsymbol{v_3}=\boldsymbol{0}
$$


를  풀면


$$
\begin{bmatrix}1 &4 & 2 &0 \\2 & 5 & 1 &0\\ 3 & 6 & 0 & 0\end{bmatrix} \sim
\begin{bmatrix}1 &4 & 2 &0 \\0 & -3 & -3 &0\\ 0 & -2 & -2 & 0\end{bmatrix} \sim
\begin{bmatrix}1 &0 & -2 &0 \\0 & 1 & 1 &0\\ 0 & 0 & 0 & 0\end{bmatrix}
$$


이 되어


$$
x_1= 2x_3 \\
x_2=-x_3 \\
x_3 : \ free \ variable
$$


이 됩니다. 즉, trivial solution이 아닌 다른 solution(nontrivial solution)이 존재하기 때문에  $\{\boldsymbol{v_1}, \boldsymbol{v_2}, \boldsymbol{v_3}\}$은 linearly dependent합니다.



<br/>



#### (1) Linear independence and Matrix column

<br/>



위의 개념을 matrix equation으로 끌고 오면, matrix column에서도 linear independence를 정의할 수 있습니다.



$m \times n $matrix $A$를 다음과 같이 정의하면


$$
A = \begin{bmatrix}\boldsymbol{a_1} & \boldsymbol{a_2} & ... & \boldsymbol{a_n}\end{bmatrix}
$$


$A\boldsymbol{x}=\boldsymbol{0}$을 다음과 같이 정의할 수 있습니다.


$$
x_1\boldsymbol{a_1}+x_2\boldsymbol{a_2}+\cdots+x_n\boldsymbol{a_n}=0
$$


Matrix eqation


$$
A\boldsymbol{x}=\boldsymbol{0}
$$


이 trivial solution만을 가질 때, **matrix** $A$**의 columns이 linearly independent**하다고 합니다.



<br/>



*example 2)*


$$
A =\begin{bmatrix}0 &1 & 4 \\1 & 2 & -1 \\ 5 & 8 & 0 \end{bmatrix}
$$


위 matrix를 이용한 matrix equation $A\boldsymbol{x}=\boldsymbol{0}$를 풀면


$$
\begin{bmatrix}0 &1 & 4 &0 \\1 & 2 & -1 &0\\ 5 & 8 & 0 & 0\end{bmatrix} \sim 
\begin{bmatrix}1 &0 & 0 &0 \\0 & 1 & 0 &0\\ 0 & 0 & 1 & 0\end{bmatrix}
$$

$$
x_1=0 \\
x_2=0 \\
x_3=0 
$$


와 같이 trivial solution만 존재합니다. 따라서 $A$의 columns은 linearly independent합니다.



<br/>



#### (2) Special Case of linearly independence



<br/>



특정 집합의 경우 linear independence를 vector equation을 풀지 않고도 바로 확인할 수 있습니다.



* Zero vector만 있는 경우


$$
\{\boldsymbol{0}\}
$$


다음의 집합의 경우 linearly dependent합니다. 이는 


$$
x_1\boldsymbol{0}=\boldsymbol{0}
$$
를 만족시키는 $x_1$이 실수 전체이기 때문입니다. 



* Zero vector가 아닌 vector 하나만 있는 경우

  

$$
\{\boldsymbol{v}\}
$$



다음 집합의 경우는 linearly independent합니다.


$$
x_1\boldsymbol{v}=\boldsymbol{0}
$$


를 만족하는 $x_1$은 0밖에 없기 때문입니다. 즉 trivial solution만을 가지므로 linearly independent합니다.



* 2개의 vector를 포함하는 경우

$$
\{\boldsymbol{v_1}, \boldsymbol{v_2}\}
$$



다음의 집합의 경우, linear independence를 판별하기 위해


$$
x_1\boldsymbol{v_1}+x_2\boldsymbol{v_2}=\boldsymbol{0}
$$


의 solution을 생각해봅시다. 만약 $\boldsymbol{v_2}$가 $\boldsymbol{v_1}$의 실수배이면, 즉 $\boldsymbol{v_2}=k\boldsymbol{v_1}$이면,


$$
x_1\boldsymbol{v_1}+x_2\boldsymbol{v_2}=(x_1+kx_2)\boldsymbol{v_1}=0
$$


이 되어, $x_1=x_2=0$이 아닌 다른 solution $x_1=-kx_2$가 존재합니다. 따라서 이 경우에는 linearly dependent합니다.



만약  $\boldsymbol{v_2}$가 $\boldsymbol{v_1}$의 실수배가 아닌 경우, 위 집합은 linearly independent합니다.



이 case를 자세히 살펴보면 independent, dependent를 용어로 사용한 이유를 알 수 있습니다. 


$$
\{\boldsymbol{v_1}, \boldsymbol{v_2}\}
$$


가 linearly dependent한 경우에는,  $\boldsymbol{v_2}$가 $\boldsymbol{v_1}$의 실수배로, 즉 $\boldsymbol{v_2}=k\boldsymbol{v_1}$로 표현할 수 있습니다. 즉, **하나의 vector를 다른 vector로 표현할 수 있습니다**(일차식으로, 또는 선형적으로). 따라서 두 벡터간에 선형 관계가 있기 때문에, independent가 아닌 dependent하다라고 말할 수 있습니다.



만약 $\boldsymbol{v_2}$가 $\boldsymbol{v_1}$의 실수배가 아닌 경우,  $\boldsymbol{v_2}$을  $\boldsymbol{v_1}$에 대한 **선형식으로 표현할 수가 없습니다.** 따라서 선형적으로 독립, independent하다고 말할 수 있습니다.



위 case를 일반적인 case로 확장한 정리가 다음 정리입니다.

<br/>



**Theorem : Characterization of Linearly Dependent Sets**





An indexed set $S = \{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ of two or more vectors is linearly dependent if and only if at least one of the vectors in $S$ is a linear combination of others.



In fact, if $S$ is linearly dependent and $\boldsymbol{v_1}\neq0$, then some $\boldsymbol{v_j}$ (with $j>1$) is a linear combination of the preceding vectors $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_{j-1}}$



정리하면, '$S$가 linearly dependent하다.'와 동치인 명제는, '$S$에 속한 적어도 하나의 vector가 나머지 vector들의 linear combination으로 표현된다.'입니다. 여기에 추가적으로, 만약 $S$ 가 linearly dependent하고 $\boldsymbol{v_1}\neq0$이면, 적어도 하나의 vector $\boldsymbol{v_j}$가 $j$보다 작은 index를 가진 vector들의 linear combination으로 표현됩니다. 즉, **linearly dependent하면, 적어도 하나의 vector가 나머지 vector들의 linear combination으로 표현됩니다. ** 



정리에 대한 증명은 밑 부분 appendix에 남겨두겠습니다. 



* vector의 개수와 성분의 수



벡터의 성분의 수보다 집합에 존재하는 벡터의 수가 더 많은 경우 그 집합은 linearly dependent합니다. 즉, $S=\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ in $\mathbb{R}^n$에서,
$$
p>n
$$


인 경우 $S$는 linearly dependent합니다. 



이 증명 또한 appendix에 남기겠습니다.



* zero vector



Zero vector를 포함한 집합은 linearly dependent합니다.





위의 special case의 경우는 굳이 linear independence를 확인하기 위해 vector equation을 풀지 않고도 linear independence를 확인할 수 있는 방법입니다.(물론 증명할 때는 정의를 이용합니다.)



지금까지 선형대수학에서 중요한 개념 중 하나인 linear independence에 대해 알아보았습니다. 다음 포스트에서는 matrix의 개념과 기본적인 matrix에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글로 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of Theorem

<br/>



#### 1) Characterization of Linearly Dependent Sets

<br/>



An indexed set $S = \{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ of two or more vectors is linearly dependent if and only if at least one of the vectors in $S$ is a linear combination of others.



In fact, if $S$ is linearly dependent and $\boldsymbol{v_1}\neq0$, then some $\boldsymbol{v_j}$ (with $j>1$) is a linear combination of the preceding vectors $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_{j-1}}$



* **proof**



위 정리에서 밝혀야 하는 명제가 두 개 입니다. 첫 번째로 



An indexed set $S = \{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ of two or more vectors is linearly dependent if and only if at least one of the vectors in $S$ is a linear combination of others.



에 대해 증명을 해보겠습니다.



(1) $\rightarrow$ 방향



 $S = \{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$가 linearly dependent하기 때문에 적어도 하나는 $0$이 아닌 $c_1, c_2, ..., c_p$가 다음을 만족합니다.


$$
c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots + c_p\boldsymbol{v_p}=0
$$


위에서 $c_j\neq0$을 만족하는 $c_j$에 대해서, $c_j\boldsymbol{v_j}$항만 우변으로 넘겨서 정리하면


$$
-c_j\boldsymbol{v_j}=c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots+c_{j-1}\boldsymbol{v_{j-1}}+c_{j+1}\boldsymbol{v_{j+1}}+\cdots+c_p\boldsymbol{v_p}
$$


가 되고, $-c_j\neq0$이므로 이를 양변에 나누어주면


$$
\boldsymbol{v_j}=-\frac{c_1}{c_j}\boldsymbol{v_1}+-\frac{c_2}{c_j}\boldsymbol{v_2}+\cdots+-\frac{c_{j-1}}{c_j}\boldsymbol{v_{j-1}}+-\frac{c_{j+1}}{c_j}\boldsymbol{v_{j+1}}+\cdots+-\frac{c_p}{c_j}\boldsymbol{v_p}
$$


다음과 같이 $\boldsymbol{v_j}$가 이를 제외한 나머지 vector들의 linear combination으로 표현됩니다.





(2) $\leftarrow$ 방향



$S$의 원소 $\boldsymbol{v_j}$이 $S$의 나머지 vector들의 linear combination으로 표현된다고 가정해봅시다.


$$
\boldsymbol{v_j}=c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots+c_{j-1}\boldsymbol{v_{j-1}}+c_{j+1}\boldsymbol{v_{j+1}}+\cdots+c_p\boldsymbol{v_p}
$$


여기서  $\boldsymbol{v_j}$를 우변으로 넘겨 정리하면


$$
c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots+c_{j-1}\boldsymbol{v_{j-1}}+\boldsymbol{v_j}+c_{j+1}\boldsymbol{v_{j+1}}+\cdots+c_p\boldsymbol{v_p}=0
$$


이 됩니다. 이 때, $c_1, ..., c_{j-1}, c_{j+1}, ..., c_p$값이 어떤 값이든 간에 위 식에서의 $c_1, ..., c_{j-1}, 1, c_{j+1}, ..., c_p$는


$$
x_1\boldsymbol{v_1}+x_2\boldsymbol{v_2}+\cdots+x_p\boldsymbol{v_p}=0
$$
의 solution이고, nontrivial solution입니다. 따라서 $S$는 linearly dependent합니다.



두 번째 명제



In fact, if $S$ is linearly dependent and $\boldsymbol{v_1}\neq0$, then some $\boldsymbol{v_j}$ (with $j>1$) is a linear combination of the preceding vectors $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_{j-1}}$



을 증명해보겠습니다. $S$가 linearly dependent하므로, 적어도 하나는 0이 아닌 $c_1, c_2, ..., c_p$가 다음을 만족합니다.


$$
c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots + c_p\boldsymbol{v_p}=0
$$


여기서 $c_i\neq0$을 만족하는 $i$ 중에서 가장 큰 index를 $j$라고 하겠습니다. 그러면


$$
c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots + c_{j-1}\boldsymbol{v_{j-1}} + c_j\boldsymbol{v_j} + 0\times\boldsymbol{v_{j+1}} + \cdots 0 \times\boldsymbol{v_p}=0
$$


와 같이 식을 표현할 수 있습니다. 이는


$$
c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots + c_{j-1}\boldsymbol{v_{j-1}} + c_j\boldsymbol{v_j}=0
$$


이 됩니다. 이제  $c_j\boldsymbol{v_j}$항만 우변으로 넘겨서 정리하면
$$
c_j\boldsymbol{v_j}=c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots + c_{j-1}\boldsymbol{v_{j-1}}
$$


$c_j\neq0$이므로 양변에 $c_j$를 나누어주면 
$$
\boldsymbol{v_j}=-\frac{c_1}{c_j}\boldsymbol{v_1}-\frac{c_2}{c_j}\boldsymbol{v_2}+\cdots-\frac{c_{j-1}}{c_j}\boldsymbol{v_{j-1}}
$$
와 같이 $\boldsymbol{v_j}$이 $\boldsymbol{v_1}, \boldsymbol{v_2}, \cdots, \boldsymbol{v_{j-1}}$의 linear combination으로 표현 가능합니다.

<br/>



#### 2) Vector의 수와 vector의 성분의 수

<br/>



벡터의 성분의 수보다 집합에 존재하는 벡터의 수가 더 많은 경우 그 집합은 linearly dependent합니다. 즉, $S=\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ in $\mathbb{R}^n$에서,


$$
p>n
$$


인 경우 $S$는 linearly dependent합니다. 



* **proof**



$S$가 linearly independent한지 아닌지 확인하기 위해 vector equation


$$
x_1\boldsymbol{v_1}+x_2\boldsymbol{v_2}+\cdots+x_p\boldsymbol{v_p}=0
$$


을 생각해봅시다. 이 vector equation의 결과와 상응하는 linear system의 augmented matrix는


$$
\begin{bmatrix} \boldsymbol{v_1} & \boldsymbol{v_2} & ... & \boldsymbol{v_p} & \boldsymbol{0} \end{bmatrix}
$$


입니다. 이 때, 이 matrix는 $n \times (p+1)$ matrix입니다.



이 matrix를 row operation을 통해서 reduced echelon form을 만들었을 때 나타나는 leading entry는 최대 $n$개입니다. ($n<p$이기 때문입니다.) 즉, 위 augmented matrix의 pivot column은 최대 $n$개가 되고, pivot column이 아닌 column이 반드시 두개 이상 존재하게 됩니다. 

따라서 위의 linear system에서 free variable이 존재하고, 이는 위의 linear system의 solution이 무수히 많은 것을 뜻합니다. 즉, trivial solution이 아닌 nontrivial solution이 존재하므로, $S$는 linearly dependent합니다.



<br/>



#### 3) Zero vector를 포함한 집합

<br/>



Zero vector를 포함한 집합은 linearly dependent합니다.





* **Proof**



$S=\{\boldsymbol{0}, \boldsymbol{v_2}, ..., \boldsymbol{p}\}$ 집합의 linear independence를 확인하기 위해 vector eqation


$$
x_1\boldsymbol{0}+x_2\boldsymbol{v_2}+\cdots+x_p\boldsymbol{v_p}=0
$$


을 생각해봅시다. 이 때, $x_1\neq0, x_2=x_3=\cdots=x_p=0$인 경우 위 equation이 성립합니다. 즉, nontrivial solution이 존재하기 때문에 $S$는 linearly dependent합니다. 



