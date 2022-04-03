---
layout: single
title:  "4.2 Linear transformation (2)"
categories: [Linear Algebra]
tag: [Linear Algebra, Linear transformation, Kernel, Range]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---





이번 포스트에서는 tranformation에서의 kernel과 range에 대해서 알아보겠습니다.



<br/>

### 1) Kernel of transformation



<br/>

#### (1) Kernel

<br/>



**Definition : Kernel of transformation** 



If $T : \mathbb R^n \rightarrow \mathbb R^m$ is a transformation, then the set of vectors in $\mathbb R^n$ that $T$ maps into $0$ is called kernel of $T$ and is denoted by $ker(T)$


$$
Ker(T) = \{\boldsymbol{x} \mid T(\boldsymbol x)=0\}
$$


즉 transformation $T$의 kernel은 $T(\boldsymbol{x})=0$을 만족시키는 모든 $\boldsymbol{x}$ 을 모은 집합입니다. 



<br/>



*example*


$$
A = \begin{bmatrix}1 & -1 \\ 2 & 5 \\ 3 & 4 \end{bmatrix}
$$


에 대해서 matrix transformation $T_A$의 kernel은


$$
T_A(\boldsymbol{x}) = A\boldsymbol{x} = 0
$$


을 만족시키는 $\boldsymbol x$의 집합니다. 따라서 
$$
\begin{bmatrix}1 & -1 & 0\\ 2 & 5  &0\\ 3 & 4 &0\end{bmatrix}
$$


다음의 augmented matrix를 가진 linear system을 푸는 문제로 바뀌고, 이를 풀게 되면


$$
\begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$


이 되어, 위 system의 solution은 zero vector밖에 존재하지 않아
$$
ker(T_A) = \{0\}
$$
가 됩니다. 



<br/>

*example*



정의역과 공역이 $\mathbb R^n$인 Zero operator $T_0$의 kernel은
$$
ker(T_0) = \{\boldsymbol{x} \mid T_0(\boldsymbol{x}) = 0\boldsymbol{x} =0\}
$$


을 만족시키는 $\boldsymbol{x}$ 집합이므로, $\mathbb R^n$에 속하는 모든 벡터가 kernel에 속합니다. 따라서
$$
ker(T_0) = \mathbb R^n
$$


이 됩니다.



<br/>

*example*



정의역과 공역이 $\mathbb R^n$인 identity operator $T_I$의 kernel은
$$
ker(T_I) = \{\boldsymbol{x} \mid T_I(\boldsymbol x) = I\boldsymbol{x} = 0\}
$$
이므로, zero vector만 성립됩니다. 따라서
$$
ker(T_I) = \{0\}
$$


가 됩니다.



Kernel과 linear transformation과 관련된 정리는 다음과 같습니다.

<br/>

**Theorem**



The kernel of a linear transformation always contains the zero vector





linear transformation의 kernel은 반드시 zero vector를 포함합니다.



<br/>

**Theorem**



If $T : \mathbb R^n \rightarrow \mathbb R^m$ is a linear transformation, then the kernel of $T$ is a subspace of $\mathbb R^n$



linear transformation의 kernel은 domain의 subspace가 됩니다. 



<br/>

#### (2) One to one

<br/>

**Definition : One to one**



The transformation $T : \mathbb R^n \rightarrow \mathbb R^m$ is one to one if $T$ maps distinct vectors in $\mathbb R^n$ into distinct vectors in $\mathbb R^m$



one to one은 함수에서 정의되는 일대일 함수의 정의와 같습니다. 즉
$$
if \ \ T(\boldsymbol{x}) = \ T(\boldsymbol{y}) \Rightarrow \boldsymbol{x} = \boldsymbol{y} \ \ for \ \ all \ \ \boldsymbol{x, y} \in \mathbb R^n
$$
일 때, $T$는 one to one이라고 합니다.  위 정의의 대우인


$$
if \ \  \boldsymbol{x} \neq \boldsymbol{y} \Rightarrow T(\boldsymbol{x}) \neq \ T(\boldsymbol{y}) \ \ for \ \ all \ \ \boldsymbol{x, y} \in \mathbb R^n
$$
또한 많이 사용됩니다. 



one to one과 linear transformation, kernel은 다음과 같은 관계를 가지고 있습니다.



<br/>

**Theorem**



If $T: \mathbb R^n \rightarrow \mathbb R^m$ is a linear transformation, then the followings are equivalent

1. $T$ is one to one
2. $ker(T) = \{\boldsymbol{0}\}$



만약 $T$가 linear transformation이면, T가 one to one임과 kernel이 zero vector만 있는 것은 동치입니다. 



<br/>

**Theorem**



If $A$ is $m \times n$ matrix, then the corresponding linear transformation $T_A : \mathbb R^n \rightarrow \mathbb R^m$ is one to one if and only if the linear system $A\boldsymbol{x} =0$ has only the trivial solution.



matrix transformation과 one to one 간의 관계를 나타내는 정리입니다. 위의 linear transformation이 one to one이면 kernel이 zero vector만을 가지는 것을 이용하면 쉽게 증명할 수 있습니다. 





<br/>

### 2) Range of transformation



<br/>

#### (1) Range of transformation

<br/>

**Definition : Range of transformation**



If $T : \mathbb R^n \rightarrow \mathbb R^m$ is a transformation, then the range(image) of $T$, denoted by $ran(T)$ is the set of all vectors in $\mathbb R^m$ that are images of at least one vector in $\mathbb R^n$



range는 함수에서 정의되는 치역과 같습니다. 즉,


$$
ran(T) = \{\boldsymbol{b} \mid \boldsymbol{b} = T(\boldsymbol{x}) \ \ for \ \ all \ \ \boldsymbol{x} \in \mathbb R^n\}
$$
kernel과 달리 range는 transfrom된 output 값들의 집합이기 때문에, 공역의 부분집합입니다.



<br/>

*example*
$$
A = \begin{bmatrix}1 & -1 \\ 2 & 5 \\ 3 & 4 \end{bmatrix}
$$


일 때, matrix transformation $T_A$의 range는


$$
ran(T_A) = \{T_A(\boldsymbol{x}) \mid \boldsymbol{x} \in \mathbb R^2\}
$$


입니다. 이 때,


$$
T_A(\boldsymbol{x}) = x_1\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + x_2\begin{bmatrix} -1 \\ 5 \\ 4 \end{bmatrix}
$$


가 되고, $\mathbb R^2$에 포함되는 모든 $\boldsymbol{x}$에 대해 위 output을 전부 모은 집합이니, 이는 $A$의 column의 linear combination을 모두 모은 집합입니다. 즉


$$
ran(T_A) = span\{\begin{bmatrix}1 \\ 2 \\ 3 \end{bmatrix}, \begin{bmatrix}-1 \\ 5 \\ 4 \end{bmatrix}\}
$$


<br/>

*example*



정의역과 공역이 $\mathbb R^n$인 zero operator의 경우 
$$
ran(T_0) = \{T_0(\boldsymbol{x}) \mid \boldsymbol x \in \mathbb R^n\} = \{0\}
$$


output이 zero vector밖에 없기 때문에, range는 zero vector만을 가집니다.



<br/>

*example*



정의역과 공역이 $\mathbb R^n$인 identity operator $T_I$의 range는
$$
ran(T_I) = \{T_I(\boldsymbol{x}) \mid \boldsymbol x \in \mathbb R^n\} = \mathbb R^n
$$
$\mathbb R^n$ 에 속한 모든 vector $\boldsymbol{x}$에 대해 output은 자기 자신이고, 이를 모두 모은 집합이니, range는 $\mathbb R^n$이 됩니다.



Range와 linear transformation은 다음과 같은 관계를 가집니다.



<br/>

**Theorem**



If $T : \mathbb R^n \rightarrow \mathbb R^m$ is a linear transformation, then $ran(T)$ is a subspace of $\mathbb R^m$



linear transformation의 range는 공역의 subspace가 됩니다.





<br/>

#### (2) Onto

<br/>

**Definition : Onto **



A transformation $T : \mathbb R^n \rightarrow \mathbb R^m$ is onto if the range of $T$ is the entire codomain $\mathbb R^m$



즉, range와 codomain이 같을 때 transformation은 onto라고 합니다.



onto와 range, linear transformation과의 관계는 다음과 같습니다. 



<br/>

**Theorem**



If $A$ is ans $m \times n$ matrix, then the corresponding linear treansformation $T_A : \mathbb R^n \rightarrow \mathbb R^m$ is onto if and only if the linear system $A\boldsymbol{x} = \boldsymbol{b}$ is consistent form every $\boldsymbol{b}$ in $\mathbb R^m$



linear transformation일 때 range의 성질과 onto의 정의를 이용하면 쉽게 증명할 수 있습니다. 





<br/>

### 3) Linear operator

<br/>

#### (1) Linear operator



앞서서 정의한 linear transformation, kernel, one to one, range, onto를 linear operator에 적용하면 다음의 정리를 얻습니다.

<br/>



**Theorem **



If $T:\mathbb R^n \rightarrow \mathbb R^n$ is a linear operator on $\mathbb R^n$, then $T$ is one to one if and only if it is onto.





<br/>

#### (2) Invertible Matrix Theorem



 linear operator에서 onto, one to one의 성질을 이용하면, linear operator의 standard matrix의 invertibility에 대해서도 논할 수 있습니다. 만약 standard matrix가 one to one이고 onto이면(linear operator에서는 동치입니다. ), 해당 standard matrix는 invertible 합니다. 해당 명제의 역 또한 성립하구요. 따라서 이전의 Invertible Matrix Theorem에 다음의 명제가 추가됩니다.



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

k. $detA\neq 0$ 

**l.** $T_A$ is one to one

**m.** $T_A$ is onto





지금까지 kernel과 range에 대해서 알아보았습니다. 다음 포스트에서는 matrix로 정의할 수 있는 vector space인 row space, column space, null space에 대해서 알아보겠습니다.  질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>

### Appendix : Proof of Theorem

<br/>



#### (1) Kernel , one to one



<br/>

**Theorem**



The kernel of a linear transformation always contains the zero vector



* **proof**



$T : \mathbb R^n \rightarrow \mathbb R^m$ 가 linear transformation일 때, $T$의 kernel은


$$
Ker(T) = \{\boldsymbol{x} \mid T(\boldsymbol x) = 0\}
$$
입니다. $T$가 linear transformation이므로


$$
T(0) = T(0  \ \boldsymbol{v}) = 0\cdot T(\boldsymbol{v})  = 0
$$


을 만족합니다. 따라서 zero vector는 linear transformation의 kernel에 반드시 포함됩니다.





<br/>

**Theorem**



If $T : \mathbb R^n \rightarrow \mathbb R^m$ is a linear transformation, then the kernel of $T$ is a subspace of $\mathbb R^n$



* **proof**



Kernel이 $\mathbb R^n$의 subspace임을 확인하기 위해서는 4가지를 확인해야 합니다. 



1. $Ker(T) \subset  \mathbb R^n$

   Kernel의 정의에 따라 $\mathbb R^n$의 subspace입니다. 

2. $0 \in Ker(T)$

   앞선 정리에서 linear transformation의 kernel에는 zero vector를 반드시 포함합니다. 

3. $\boldsymbol{u, v} \in Ker(T) \Rightarrow T(\boldsymbol{u})=T(\boldsymbol{v}) = 0 $

   이므로, $T(\boldsymbol{u+v}) = T(\boldsymbol{u}) + T(\boldsymbol{v}) = 0 $을 만족하여 $\boldsymbol{u+v} \in Ker(T)$을 만족합니다.

4. $\boldsymbol{u} \in Ker(T), c \in \mathbb R $

   일 때, $T(c\boldsymbol{u}) = cT(\boldsymbol u) = 0$이 되어 $c\boldsymbol u \in Ker(T)$를 만족합니다. 



따라서 subspace의 조건을 모두 만족하므로 $Ker(T)$는 $\mathbb R^n$의 subspace입니다.



<br/>

**Theorem**



If $T: \mathbb R^n \rightarrow \mathbb R^m$ is a linear transformation, then the followings are equivalent

1. $T$ is one to one
2. $ker(T) = \{\boldsymbol{0}\}$



* **proof**



$T$ is one to one $\Rightarrow Ker(T) = \{0\}$



$T$가 one to one이면


$$
T(\boldsymbol{x}) = T(\boldsymbol{y}) \Rightarrow \boldsymbol{x} = \boldsymbol y
$$


을 만족합니다. 또한 $T$가 linear transformation이므로 $0$는 $T$의 kernel에 속합니다. 만약 $0$ 가 아닌 다른 벡터 $\boldsymbol{v}$가 $Ker(T)$에 속한다고 가정해봅시다.

그럼


$$
T(\boldsymbol{v}) = T(0) = 0
$$


이고, $T$가 one to one이므로


$$
\boldsymbol{v} = 0
$$


가 됩니다. 현재 $\boldsymbol{v}$는 $0$가 아니라고 가정하였기 때문에 모순이 발생하여, $ker(T)=\{0\}$입니다. 



$Ker(T) = \{0\} \Rightarrow $ $T$ is one to one 



$T$가 one to one임을 확인하기 위하여


$$
T(\boldsymbol{x}) = T(\boldsymbol{y})
$$


인 경우를 생각해봅시다. 이는 $T$가 linear transformation이므로


$$
T(\boldsymbol{x}) - T(\boldsymbol{y}) = T(\boldsymbol{x-y})=0
$$
가 성립합니다. 즉 $\boldsymbol{x-y} \in Ker(T)$이고, $Ker(T)$에 속한 vector는 $0$이므로


$$
\boldsymbol{x-y}=0 \Rightarrow \boldsymbol{x}=\boldsymbol{y}
$$


따라서 $T$는 one to one이 됩니다.



<br/>

**Theorem**



If $A$ is $m \times n$ matrix, then the corresponding linear transformation $T_A : \mathbb R^n \rightarrow \mathbb R^m$ is one to one if and only if the linear system $A\boldsymbol{x} =0$ has only the trivial solution.



* **proof**



$T_A$가 one to one이면 $T_A$의 kernel은 zero vector만 존재합니다. 즉


$$
Ker(T_A) = \{\boldsymbol{x} \mid T_A(\boldsymbol{x}) = A\boldsymbol{x}=0\} = \{0\}
$$
을 만족시키는 $\boldsymbol{x}$가 $0$밖에 없기 때문에, linear system $A\boldsymbol{x} =0$ 은 trivial solution만을 가지게 됩니다. 



반대로 linear system $A\boldsymbol{x} =0$ 이 trivial solution만을 가지면 $T_A$의 kernel이 zero vector만 가지기 때문에, one to one이 성립됩니다. 





<br/>



#### (2) Range, onto



<br/>

**Theorem**





If $T : \mathbb R^n \rightarrow \mathbb R^m$ is a linear transformation, then $ran(T)$ is a subspace of $\mathbb R^m$



* **proof**



$T$의 range가 $\mathbb R^m$의 subspace임을 밝히기 위해서는 4가지를 확인해야 합니다.



1. $ran(T) \subset \mathbb R^m$

   range의 정의에 의해 성립합니다.

2. $0 \in ran(T)$

   $T$가 linear transformation이므로 $T(0)=0$임을 만족하고, 따라서 $0 \in ran(T)$를 만족합니다.

3. $\boldsymbol{u, v} \in ran(T)$ 이면, 어떤 $\boldsymbol{x, y} \in \mathbb R^n$이 존재하여 $T(\boldsymbol{x})=\boldsymbol{u},T(\boldsymbol{y})=\boldsymbol{v}$을 만족합니다. 이 때

   $\boldsymbol{u+v}= T(\boldsymbol{x}) + T(\boldsymbol{y}) = T(\boldsymbol{x+y}) \in ran(T)$입니다.

4. $\boldsymbol{u} \in ran(T), c \in \mathbb R$에 대해서,  $T(\boldsymbol{x}) = \boldsymbol{u}$을 만족하는 $\boldsymbol{x}$가 존재합니다. 따라서

   $c\boldsymbol{u} = cT(\boldsymbol{x}) = T(c\boldsymbol{x}) \in ran(T)$ 입니다.



따라서 $ran(T)$는 $\mathbb R^m$의 subspace입니다. 



<br/>

**Theorem**



If $A$ is ans $m \times n$ matrix, then the corresponding linear treansformation $T_A : \mathbb R^n \rightarrow \mathbb R^m$ is onto if and only if the linear system $A\boldsymbol{x} = \boldsymbol{b}$ is consistent for every $\boldsymbol{b}$ in $\mathbb R^m$



* **proof**



 $T_A : \mathbb R^n \rightarrow \mathbb R^m$ 가 onto이므로, $ran(T_A) = \mathbb R^m$입니다. range의 정의가


$$
ran(T_A) = \{T_A(\boldsymbol{x}) \mid \boldsymbol{x} \in \mathbb R^n\} = \mathbb R^m
$$
이므로, $\mathbb R^m$에 존재하는 임의의 vector $\boldsymbol b$에 대해 $T_A(\boldsymbol x) = A\boldsymbol x$를 만족하는 $\boldsymbol x$존재합니다. 즉


$$
A\boldsymbol x = \boldsymbol b
$$


는 모든 $\boldsymbol b$에 대해 solution을 가집니다. 즉 consistent합니다. 



마찬가지로, $A\boldsymbol{x} =\boldsymbol{b}$가 모든 $\boldsymbol b\in \mathbb R^m$에 대해 consistent하면 range 정의에 따라 $ran(T_A)=\mathbb R^m$입니다. 따라서 $T_A$는 onto입니다.



<br/>

#### (3) Linear operator



<br/>



**Theorem **



If $T:\mathbb R^n \rightarrow \mathbb R^n$ is a linear operator on $\mathbb R^n$, then $T$ is one to one if and only if it is onto.



* **proof**



$T$가 one to one이면  $T$의 standard matrix $[T]$에 대해서


$$
[T]\boldsymbol x = 0
$$


이 반드시 trivial solution을 가집니다. 이는 invertible matrix theorem에 따라 $[T]$는 invertible합니다.

$[T]$가 invertible하면, $\mathbb R^n$에 속하는 모든 $\boldsymbol b$에 대해


$$
[T]\boldsymbol{x} = \boldsymbol b
$$
 

가 consistent합니다. 따라서 


$$
ran(T) = \mathbb R^n
$$


이므로 $T$는 onto입니다.



위 과정을 거꾸로 진행하면 $T$가 onto이면 $T$가 one to one인 것 또한 쉽게 알 수 있습니다.
