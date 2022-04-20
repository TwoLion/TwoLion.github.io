---
layout: single
title:  "6.1 Inner Product, Length, Orthogonality"
categories: [Linear Algebra]
tag: [Linear Algebra]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---







이번 포스트에서는 inner product, length, orthogonality에 대해서 알아보도록 하겠습니다.



<br/>



### 1) Inner Product



<br/>



$\mathbb R^n$에 속하는 두 벡터 $\boldsymbol{v}, \boldsymbol{u}$ 간의 inner product는 다음과 같이 정의됩니다. 



<br/>

**Definition : Inner product**



If $\boldsymbol v, \boldsymbol u \in \mathbb R^n$, such that


$$
\boldsymbol{u} =\begin{bmatrix}u_1 \\ \vdots \\ u_n\end{bmatrix}, \ \ \boldsymbol{v} = \begin{bmatrix}v_1 \\ \vdots \\ v_n \end{bmatrix}
$$


The number $\boldsymbol v ^T \boldsymbol u$ is called the inner product of $\boldsymbol{u}$ and $\boldsymbol{v}$. 


$$
\boldsymbol u \cdot \boldsymbol v = \boldsymbol v^T \boldsymbol u = u_1v_1+\cdots +u_nv_n = \Sigma_{i=1}^nu_iv_i
$$






즉 각각 벡터의 같은 위치의 성분을 더하고 합한 것을 두 벡터의 inner product, 내적이라고 합니다. 정의에서 알 수 있듯이 같은 공간 $\mathbb R^n$에 있는 두 벡터 간의 연산입니다.





<br/>

*example*


$$
\boldsymbol{u} =\begin{bmatrix}2 \\ 1 \\ 3\end{bmatrix}, \ \ \boldsymbol{v} = \begin{bmatrix}4 \\ 2 \\ -1 \end{bmatrix}
$$


일 때


$$
\boldsymbol{u}\cdot \boldsymbol{v} = \boldsymbol{v}^T\boldsymbol u = 2\times4 + 1\times 2 + 3\times (-1) = 7
$$


<br/>



#### (1) Property of the inner product



<br/>



Inner product에 대한 성질은 다음과 같습니다. 



<br/>

**Property of inner product**



If $\boldsymbol{u, v, w} \in \mathbb R^n $ and $k \in \mathbb R$



1. $\boldsymbol{u}\cdot\boldsymbol{v} = \boldsymbol{v}\cdot\boldsymbol{u}$
2. $(\boldsymbol{u}+\boldsymbol{v})\cdot\boldsymbol{w} = \boldsymbol{u}\cdot\boldsymbol{w}+\boldsymbol{v}\cdot\boldsymbol{w}$
3. $(c\boldsymbol{u})\cdot \boldsymbol{v} = c(\boldsymbol{u}\cdot \boldsymbol{v})$
4. $\boldsymbol{u}\cdot\boldsymbol{u}\geq 0$, $\boldsymbol{u}\cdot \boldsymbol{u}=0$ if and only if $\boldsymbol{u}=0$





 inner product는 교환법칙이 성립합니다. 두 번째로 분배 법칙 또한 성립합니다. 세 번째는, scalar multiple에 대해서 inner product와의 연산 순서를 바꾸어도 결과는 변하지 않습니다. 마지막 성질이 inner product의 중요한 성질 중 하나인데, **같은 벡터의 inner product는 항상 0보다 크고, inner product 값이 0이 나오는 벡터는 zero vector만 존재합니다.**



<br/>

### 2) The length of a vector



<br/>



**Definition : Length**



The length (or norm, 크기) of $\boldsymbol v$  is the non negative scalar.


$$
\|\boldsymbol v \| = \sqrt{v_1^2 + \cdots + v_n^2}, \ \ \|\boldsymbol{v}\|^2 = \boldsymbol{v} \cdot \boldsymbol{v}
$$




어떤 벡터의 length, norm, 크기는 해당 벡터의 inner product에 루트를 취한 값으로 정의합니다. 



여기서 **vector의 length가 1인 벡터를 unit vector라고 합니다.**





<br/>

*example*


$$
\boldsymbol{v} = (1, -2, 2, 0)
$$


일 때


$$
\|\boldsymbol{v}\| = \sqrt{\boldsymbol v \cdot \boldsymbol v} = \sqrt{1 + 4 + 4}  =3
$$




<br/>

#### (1) Properties of length

<br/>

length에 대한 성질은 다음과 같습니다.



<br/>

**Property of length**



Let $\boldsymbol v \in \mathbb R^n, \ \ k\in \mathbb R$



1. $\|c\boldsymbol{v}\| = \begin{vmatrix}c\end{vmatrix}\|\boldsymbol{v}\|$
2. Normalizing : devide a nonzero vector $\boldsymbol{v}$ by its length - unit vector $\frac{1}{\|\boldsymbol v\|}\boldsymbol v$





첫 번째로 scalr multiple한 벡터의 length는 원래 벡터의 length에 scalar의 절댓값을 취한 값을 곱한 값으로 결과가 나옵니다. 두 번째는, 임의의 nonzero 벡터를 그 벡터의 length로 나누어서 unit vector로 만들 수 있습니다. normalizing을 한 벡터는 원래 벡터와 방향은 같지만 length는 1인 unit vector가 됩니다.





<br/>

### 3) Distance

<br/>



**Definition : Distance**



If $\boldsymbol{u, v} \in \mathbb R^n$, the distance between $\boldsymbol{u}$ and $\boldsymbol v$ is the length of the vector $\boldsymbol{u-v}$


$$
dist(\boldsymbol{u}, \boldsymbol{v}) = \|\boldsymbol{u-v}\|
$$


두 벡터 사이의 거리는 두 벡터의 차의 크기로 정의합니다.

$\mathbb R^2$의 경우, 우리가 알고 있는 두 점 사이의 거리와 결과가 같습니다. 이전 벡터에 대해서 설명할 때, 시점을 원점으로 고정시키면 벡터와 점은 일대일 대응이 가능하다고 하였습니다. 이를 통해 두 벡터간 거리와 점과 점 사이의 거리 식이 같은 것을 알 수 있습니다. 


$$
\boldsymbol{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \ \ \boldsymbol{y} = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} 
$$


의 경우 위 벡터를 점으로 보면, 두 점 사이의 거리는


$$
\sqrt{(x_1-y_1)^2 + (x_2-y_2)^2} = \sqrt{\|\boldsymbol{x} -\boldsymbol{y}\|^2} = \|\boldsymbol{x-y}\|
$$


인 것을 알 수 있습니다. 





<br/>

### 4) Orthogonal Vectors

<br/>



**Definition : Orthogonal vectors**



Two vectors $\boldsymbol{u, v} \in \mathbb R^n$ are orthogonal if


$$
\boldsymbol{u}\cdot \boldsymbol{v} =0 \iff \boldsymbol{u}\perp\boldsymbol{v}
$$


The zero vector is orthogonal to every vector in $\mathbb R^n$





두 벡터가 orthogonal(직교, 수직)이라는 것은 두 벡터의 inner product값이 0이 되는 것을 뜻합니다. 



<br/>

**Theorem**



Two vectors $\boldsymbol{u, v}$ are orthogonal if and only if 


$$
\|\boldsymbol{u+v}\|^2 =\|\boldsymbol{u}\|^2 + \|\boldsymbol{v}\|^2
$$




orthogonal의 정의를 이용하면 쉽게 밝힐 수 있습니다. 





<br/>

### 5) Orthogonal complements

<br/>



Supspace에서 orthogonal 개념을 이용하여 새로운 집합인 orthogonal complement을 정의할 수 있습니다. 



<br/>

**Definition : Orthogonal Complements**



If a vector $\boldsymbol{z}$ is orthogonal to every vector in a subspace $W$ of $\mathbb R^n$, then $\boldsymbol{z}$ is orthogonal to $W$.

The Set of all vectors $\boldsymbol{z}$ that are orthogonal to $W$ is orthogonal complement of $W$


$$
W^\perp = \{\boldsymbol{z} \mid \boldsymbol{z}\cdot\boldsymbol{w} = 0 \ \ for \ \ all \ \ \boldsymbol{w} \in W \}
$$




벡터 $\boldsymbol{z}$가 subspace $W$에 있는 모든 벡터와 orthogonal하면, $\boldsymbol{z}$는 $W$에 orthogonal합니다. 이 때, $W$에 orthogonal한 모든 벡터를 모은 집합을 $W$의 orthogonal complement라고 합니다.

아래의 정리들을 통해 orthogonal complement의 특징을 확인할 수 있습니다.







<br/>

**Theorem**



A vector $\boldsymbol x$ is in $W^\perp$ if and only if $\boldsymbol{x}$ is orthogonal to every vector in a set that spans $W$



$\boldsymbol{x}$가 $W$에 orthogonal하려면 $W$에 있는 모든 벡터와 $\boldsymbol{x}$가 orthogonal한지 확인해야 하지만, 위 정리는 $W$를 span하는 벡터들만 확인하여 orthogonality를 파악할 수 있습니다. Subspace $W$에 속한 벡터가 너무 많다면, 이를 잘 설명할 수 있는, basis를 생각할 수 있는데, basis 조건 중 하나가 basis로 $W$를 span해야하는 조건입니다. **위 정리를 통해  $W$의 basis 벡터와 orthogonal한지 확인함으로써 어떤 벡터가 $W$에 orthogonal한지 파악할 수 있습니다.**



<br/>



**Theorem**



$W^\perp$ is a subspace of $\mathbb R^n$



$W$의 orthogonal complement 또한 subspace입니다. 



<br/>

**Theorem**



Let $A$ be an $m \times n $ matrix. The orthogonal complement of the row space of $A$ is the null space of $A$

The orthogonal complement of the column space of $A$ is the null space of $A^T$


$$
(RowA)^\perp = NulA, \ \ (ColA)^\perp = NulA^T
$$


matrix로 정의되는 subspace간 orthogonal 관계를 설명한 정리입니다. **Row space의 orthogonal complement는 null space가 됩니다.** $ColA$는 $RowA^T$와 같다는 점을 이용하면 $(ColA)^\perp = NulA^T$인 것을 쉽게 확인할 수 있습니다.



<br/>



**Theorem**



Let $W$ be a subspace of $\mathbb R^n$. Then


$$
\dim W +\dim W^\perp = n 
$$


어떤 subspace $W$의 dimension과 $W$의 orthogonal complement의 dimension의 합은 둘을 포함하는 vector space($\mathbb R^n$)의 diemension이 됩니다.



<br/>

지금까지 inner product, length, distance, orthogonality, orthogonal complement에 대해 알아보았습니다. 다음 포스트에서는 orthogonal set에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of Theorem



<br/>

**Theorem**



Two vectors $\boldsymbol{u, v}$ are orthogonal if and only if 


$$
\|\boldsymbol{u+v}\|^2 =\|\boldsymbol{u}\|^2 + \|\boldsymbol{v}\|^2
$$
<br/>

* **Proof**


$$
\begin{aligned}

\|\boldsymbol{u+v}\|^2  &= (\boldsymbol{u+v})\cdot(\boldsymbol{u+v}) \\
&= \|\boldsymbol{u}\|^2 + \boldsymbol{u}\cdot\boldsymbol{v} +\boldsymbol{v}\cdot\boldsymbol{u} + \|\boldsymbol{v}\|^2 \\
&=\|\boldsymbol{u}\|^2 + 2\boldsymbol{u}\cdot\boldsymbol{v}+ \|\boldsymbol{v}\|^2 \\
&= \|\boldsymbol{u}\|^2+ \|\boldsymbol{v}\|^2

\end{aligned}
$$


이는


$$
\boldsymbol u \cdot \boldsymbol v = 0
$$


을 뜻하므로, 두 벡터가 orthogonal한 것을 뜻합니다.



<br/>

**Theorem**



A vector $\boldsymbol x$ is in $W^\perp$ if and only if $\boldsymbol{x}$ is orthogonal to every vector in a set that spans $W$



<br/>

* **Proof**



<br/>



Proof of  $\Rightarrow$



$\boldsymbol{x}$가 $W^\perp$에 속하므로, $W$에 속한 모든 vector와 orthogonal합니다. 즉, $W$를 span하는 vector 역시 $W$에 포함하므로, $\boldsymbol{x}$는 $W$를 span하는 set에 속한 vector와 orthogonal합니다.



<br/>



Proof of $\Leftarrow$






$$
W =Span{S}
$$


을 만족하는 $S=\{\boldsymbol{v_1}, ..., \boldsymbol{v_p}\}$에 대해서


$$
\boldsymbol{v_j} \perp \boldsymbol{x}, j=1,...,p
$$


를 만족합니다. 이 때, $\boldsymbol{y} \in W$인 $\boldsymbol{y}$는 다음과 같이


$$
\boldsymbol{y}=c_1\boldsymbol{v_1}+\cdots+c_p\boldsymbol{v_p}
$$


표현할 수 있습니다. 해당 벡터와 $\boldsymbol{x}$를 inner product 연산을 하면


$$
\begin{aligned}

\boldsymbol{x}\cdot\boldsymbol{y}&=\boldsymbol{x}\cdot(c_1\boldsymbol{v_1}+\cdots+\boldsymbol{v_p})  \\
&=c_1\boldsymbol{x}\cdot\boldsymbol{v_1}+\cdots + c_p\boldsymbol{x}\cdot\boldsymbol{v_p} \\
&=0

\end{aligned}
$$


이 되므로, $W$에 속한 모든 벡터에 대해서 orthogonal하므로


$$
\boldsymbol{x} \in W^\perp
$$


가 성립합니다.



<br/>



**Theorem**



$W^\perp$ is a subspace of $\mathbb R^n$

<br/>

* **Proof**


$$
W^\perp = \{\boldsymbol{z} \mid \boldsymbol{z}\cdot\boldsymbol{w} = 0 \ \ for \ \ all \ \ \boldsymbol{w} \in W \}
$$


이므로, $\boldsymbol{z} \in W^\perp \subset \mathbb R^n$을 만족합니다. 



1.  zero vector는 $\mathbb R^n$에 속한 모든 벡터와 orthogonal하므로 $W$에 속한 모든 벡터와 orthogonal합니다. 따라서 $0\in W^\perp$
2. $\boldsymbol{u, v} \in W^\perp$이면, 모든 $y\in W$에 대해서 $\boldsymbol{u}\cdot\boldsymbol{y}=0, \boldsymbol{v}\cdot\boldsymbol{y}=0$임을 만족합니다. 이 때 모든 $\boldsymbol{y} \in W$에 대해서


$$
(\boldsymbol{u}+\boldsymbol{v})\cdot \boldsymbol{y} = \boldsymbol{u}\cdot \boldsymbol{y} + \boldsymbol{v} \cdot \boldsymbol{y} = 0
$$


을 만족하므로, $\boldsymbol{u+v} \in W^\perp$입니다.



3. $\boldsymbol{u} \in W^\perp$이고, scalar $k$, 임의의 $\boldsymbol y \in W$에 대해


$$
(k\boldsymbol{u})\cdot \boldsymbol{y} = k(\boldsymbol{u}\cdot \boldsymbol{y}) =0
$$


을 만족하므로, $k\boldsymbol{u} \in W^\perp$입니다.



따라서, subspace의 조건을 모두 만족하였기 때문에, $W^\perp$는 $\mathbb R^n$의 subspace입니다.



<br/>

**Theorem**



Let $A$ be an $m \times n $ matrix. The orthogonal complement of the row space of $A$ is the null space of $A$

The orthogonal complement of the column space of $A$ is the null space of $A^T$


$$
(RowA)^\perp = NulA, \ \ (ColA)^\perp = NulA^T
$$
<br/>

* **Proof**



$RowA$의 orthogonal complement는 다음의 조건을 만족해야합니다.


$$
(RowA)^\perp = \{\boldsymbol{x} \in \mathbb R^n \mid \boldsymbol{x}\cdot \boldsymbol{y} = 0 \ for \ \ all \ \ \boldsymbol{y} \in RowA \}
$$


$\boldsymbol{y}$는 $A$의 row의 linear combination으로 표현됩니다. matrix A를


$$
A = \begin{bmatrix} -\boldsymbol{a_1}- \\ \vdots \\ -\boldsymbol{a_m}- \end{bmatrix}
$$


으로 표현하면($\boldsymbol{a_i}\in \mathbb R^n$),


$$
\boldsymbol{a_i}\cdot\boldsymbol{x} =0
$$


을 만족해야 하고, 이는 


$$
\begin{bmatrix} -\boldsymbol{a_1}- \\ \vdots \\ -\boldsymbol{a_m}- \end{bmatrix}\boldsymbol{x} = \begin{bmatrix} \boldsymbol{a_1}\cdot\boldsymbol{x}  \\ \vdots \\ \boldsymbol{a_m}\cdot\boldsymbol{x} \end{bmatrix} = 0
$$


임을 뜻합니다. 이는


$$
A\boldsymbol{x}=0
$$


을 만족하므로, $\boldsymbol{x} \in NulA$입니다. 



$ColA = RowA^T$이므로, $(ColA)^\perp = (RowA^T)^\perp = NulA^T$가 성립합니다.



<br/>



**Theorem**



Let $W$ be a subspace of $\mathbb R^n$. Then


$$
\dim W +\dim W^\perp = n
$$


<br/>



* **Proof**



$W$의 basis를 $B_W = \{\boldsymbol{v_1}, ..., \boldsymbol{v_k}\}$, $W^\perp$의 basis를 $B_{W^\perp}=\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$라 하였을 때


$$
B_W \cup B_{W^\perp} = \{\boldsymbol{v_1}, ..., \boldsymbol{v_k}, \boldsymbol{u_1}, ..., \boldsymbol{u_p}\}
$$


가 $\mathbb R^n$의 basis가 되는 것을 밝히면 됩니다. 



$\mathbb R^n$에 속하는 vector $\boldsymbol x$는 다음과 같이 표현할 수 있습니다. 


$$
\boldsymbol{x} = \boldsymbol{x_1} + \boldsymbol{x_2} , \\ where \ \ \boldsymbol{x_1} \in W, \ \ \boldsymbol{x_2}\in W^\perp
$$


(이 후 projection의 개념을 알게 되면, 모든 벡터는 다음과 같이 어떤 subspace의 벡터와 subspace의 orthogonal complement에 속한 벡터의 합으로 표현이 가능한 것을 이해할 수 있습니다. )



$\boldsymbol{x_1} \in W$이므로 


$$
\boldsymbol{x_1}=c_1\boldsymbol{v_1}+\cdots +c_k \boldsymbol{v_k}
$$


가 되고, $\boldsymbol{x_2} \in W^\perp$이므로 


$$
\boldsymbol{x_2} = d_1\boldsymbol{u_1}+\cdots + d_k \boldsymbol{u_p}
$$


가 되므로


$$
\boldsymbol{x} = c_1\boldsymbol{v_1} +\cdots + c_k\boldsymbol{v_k} + d_1\boldsymbol{u_1} +\cdots +d_p\boldsymbol{u_p}
$$


임을 알 수 있습니다. 즉 $B_W \cup B_{W^\perp}$는 $\mathbb R^n$을 span합니다. 

다음으로, $B_W\cup B_{W^\perp}$가 linearly independent함을 밝혀야 합니다. 이를 밝히기 위해 다음 equation


$$
c_1\boldsymbol{v_1}+\cdots c_k\boldsymbol{v_k} + d_1\boldsymbol{u_1}+\cdots + d_p\boldsymbol{u_p}=0
$$


이 trivial solution을 가짐을 밝혀야 합니다. 먼저 $B_W, B_{W^\perp}$는 basis이므로 각각 linearly independent합니다. 따라서 만약의 위의 식이 non-trivial solution을 가진다면 $c_1,... ,c_k$ 중 적어도 하나가 0이 아니고, $d_1, ..., d_p$ 중 적어도 하나가 0이 아니어야 합니다. 이 말은 즉, 특정 벡터 $v_j$가 $B_{W^\perp}$의 linear combination으로 표현이 가능하나는 것을 뜻합니다.(또는 $\boldsymbol{u_j}$가 $B_W$의 linear combination으로 표현된다는 것을 뜻합니다. )

하지만,


$$
W\cap W^\perp = \{0\}
$$


를 만족하기 때문에, $\boldsymbol{v_j}$가 $B_{W^\perp}$의 linear combination으로 표현되거나, $\boldsymbol{u_j}$가 $B_W$의 linear combination으로 표현될 수 없습니다. 따라서 $B_W \cup B_{W^\perp}$는 linearly independent합니다. 



($W\cap W^\perp = \{0\}$인 이유는, $\boldsymbol{x} \in W, \boldsymbol{x} \in W^\perp$이면, $\boldsymbol{x} \cdot \boldsymbol{x} = 0$을 만족해야 하는데, 이를 만족하는 $\boldsymbol{x}$는 zero vector밖에 없기 때문입니다.)





