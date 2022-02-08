---
layout: single
title:  "4.1 Vector Space"
categories: [Linear Algebra]
tag: [Linear Algebra, Vector Space]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---







이번 포스트에서는 Vector space와 subspace에 대해서 알아보겠습니다.

<br/>



### 1) Vector Space



<br/>



#### 1) Vector Space



<br/>



이전까지 어떤 집합에서 연산을 다룰 때 집합 간의 연산만을 다루었지(ex : 합집합, 교집합, 차집합 등등...), 집합 내의 원소간 연산은 다루지 않았습니다. 집합에서 집합 내의 원소간 연산을 추가하여 vector space를 정의합니다. 



<br/>



**Definition : Vector space**



A vector space is a nonempty set $V$ of objects, called vectors, on which are defined two operation, called **addition** and **scalar multiplication(real numbers)**, subject to ten axioms listed below. The axioms must hold for all vectors $\boldsymbol{u}$ and $\boldsymbol{v}$ in $V$ and for all scalars $c$ and $d$



1. The sum of $\boldsymbol{u}$ and $\boldsymbol{v}$, denoted by $\boldsymbol{u+v}$, is in $V$
2. $\boldsymbol{u+v}=\boldsymbol{v+u}$
3. $\boldsymbol{(u+v)+w = u+(v+w)}$
4. There is a **zero vector ** $\boldsymbol{0}$ in $V$ such that $\boldsymbol{0+u}=\boldsymbol{u}$
5. For each $\boldsymbol{u}$ in $V$, there is a vector $\boldsymbol{-u}$ in $V$ such that $\boldsymbol{u+(-u)=0}$
6. The scalar multiple of $\boldsymbol{u}$ by $c$, denoted by $c\boldsymbol{u}$ is in $V$
7. $c(\boldsymbol{u+v}) = c\boldsymbol{u}+c\boldsymbol{v}$ 
8. $(c+d)\boldsymbol{u}=c\boldsymbol{u} + d\boldsymbol{u}$
9. $c(d\boldsymbol{u})=(cd)\boldsymbol{u}$
10. $1\boldsymbol{u}=\boldsymbol{u}$



Vector space에 속한 원소들을 vector라고 하며, 추가적으로 두 개의 연산이 정의됩니다. 한 연사는 **addition**, 다른 연산은 **scalar multiplication**입니다. vector space에서 두 연산이 정의가 되기 위해서는 10가지의 공리를 만족해야 합니다. 따라서 위의 10가지 공리를 만족하는 집합 $V$를 **vector space**라고 합니다.



위 공리에서 특히 중요한 공리는 다음 3개 입니다. 



1. There is a **zero vector ** $\boldsymbol{0}$ in $V$ such that $\boldsymbol{0+u}=\boldsymbol{u}$
2. The sum of $\boldsymbol{u}$ and $\boldsymbol{v}$, denoted by $\boldsymbol{u+v}$, is in $V$
3. The scalar multiple of $\boldsymbol{u}$ by $c$, denoted by $c\boldsymbol{u}$ is in $V$



첫 번째는 zero vector 유무입니다. Vector space는 반드시 **zero vector**를 포함합니다.

두 번째는 **'덧셈에 대해 닫혀있다'**입니다. 즉 $V$에 있는 임의의 두 vector의 합 또한 $V$에 존재해야 합니다.

세 번째는 **'scalar multiplication에 닫혀있다'**입니다. 즉 $V$에 있는 임의의 vector와 임의의 scalar에 대해서, scalar multiplication 결과 역시 $V$에 존재해야 합니다. 



이 세 가지 조건을 만족하면, vector space를 정의할 때의 10가지 공리를 모두 만족하게 됩니다. 따라서 어떤 set이 vector space임을 확인할 때 위 세 조건을 만족 여부를 통해 확인하게 됩니다.



<br/>



*example*



The space $\mathbb{R}^3$ : vector space



$\mathbb{R}^3$은 vector space입니다.


$$
\mathbb{R}^3 = \{\begin{bmatrix}x_1 \\ x_2 \\ x_3\end{bmatrix} \mid x_1, x_2, x_3 \in \mathbb R\}
$$


1. zero vector 


$$
\boldsymbol{0} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} \in \mathbb R^3
$$


zero vector는 $\mathbb R^3$에 존재합니다.



2. Addition


$$
\boldsymbol{u, v} \in \mathbb R^3 \\

\boldsymbol{u} =\begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix}, \boldsymbol{v} =\begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} \\

\boldsymbol{u+v} =\begin{bmatrix} u_1+v_1 \\ u_2+v_2 \\ u_3+v_3 \end{bmatrix} \in \mathbb R^3
$$


$\mathbb R^3$에 있는 두 벡터의 합 또한 $\mathbb R^3$에 존재합니다. 따라서 덧셈에 대해 닫혀있습니다.



3. Scalar multiplication


$$
\boldsymbol{u} \in \mathbb R^3, k \in \mathbb R \\

\boldsymbol{u} =\begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix} \\

k\boldsymbol{u} =\begin{bmatrix} ku_1 \\ ku_2 \\ ku_3 \end{bmatrix} \in \mathbb R^3
$$


 $\mathbb R^3$에 있는 벡터의 scalar multiple 값 또한  $\mathbb R^3$에 존재합니다. 따라서 scalar multiplication에 대해 닫혀있습니다.

따라서  $\mathbb R^3$는 vector space입니다.



이를 확장하면, **$\mathbb R^n$ 또한 vector space임을 알 수 있습니다.**



<br/>



*example*



For $n\geq 0$, the set $\mathbb P_n$ of polynomials of degree at most $n$ consists of all polynomials of the form


$$
\boldsymbol{p}(t)=a_0+a_1t+\cdots+a_nt^n
$$


where all the coefficients $a_0, a_1, ..., a_n$ and the variable $t$ are real numbers. If all of coefficients $a_0, ..., a_n$ are zero, this polynomial is called zero polynomial



차수가 $n$보다 작거나 같은 모든 다항식을 모은 집합을 $\mathbb P_n$이라 합시다. 이 때, $\mathbb P_n$ 역시 vector space가 됩니다.



1. zero vector


$$
a_0=a_1=...=a_n=0
$$


모든 coefficient가 0일 때, zero polynomial이 되고, zero polynomial이 zero vector의 역할을 합니다. (zero polynomial에 어떤 polynomial을 더하든 자기 자신이 나오기 때문이죠.)



2. Addition


$$
\boldsymbol{p}(t)=a_0+a_1t+\cdots+a_nt^n \\
\boldsymbol{q}(t)=b_0+b_1t+\cdots+b_nt^n
$$


라고 했을 때, 두 다항식의 합


$$
\begin{aligned}

\boldsymbol{p}(t)+\boldsymbol{q}(t)&=(a_0+a_1t+\cdots+a_nt^n) + (b_0+b_1t+\cdots+b_nt^n)\\
&=(a_0+b_0) + (a_1+b_1)t + \cdots + (a_n + b_n)t^n

\end{aligned}
$$


또한 $\mathbb P_n$에 속합니다. 결과의 coefficient가 실수이기 때문입니다. 





3. Scalar multiplication


$$
\boldsymbol{p}(t)=a_0+a_1t+\cdots+a_nt^n , \ k \in \mathbb R
$$


일 때,


$$
k\boldsymbol{p}(t)=k(a_0+a_1t+\cdots+a_nt^n) = ka_0+ka_1t+\cdots+ka_nt^n  
$$


가 되고, 이 역시 $\mathbb P_n$에 속합니다.



위 세 가지 조건을 모두 만족하기 때문에, $\mathbb P_n$ 역시 vector space가 됩니다.



<br/>



#### 2) Subspace



<br/>



집합에도 부분집합이 있듯이, vector space 역시 부분집합과 같은 개념인 subspace가 존재합니다.





<br/>



**Definition : Subspace**



A subspace of a vector space $V$ is a subset $H$ of $V$ that has three properties



1. The zero vector of $V$ is in $H$
2. $H$ is closed under vector addition. For each $\boldsymbol{u}, \boldsymbol{v}$ in $H$, the sum $\boldsymbol{u+v}$ is in $H$
3. $H$ is closed under multiplication by scalars. For each $\boldsymbol{u}$ in $H$ and each scalar $c$, the vector $c\boldsymbol{u}$ is in $H$



즉 subspace는 **어떤 vector space의 부분집합이면서, vector space의 조건을 만족하는 집합**입니다. Subspace가 되기 위한 조건은 총 4가지로



1. $H$는 $V$의 부분집합이다.
2. Zero vector가 $H$에 존재해야 한다.
3. $H$는 덧셈에 대해 닫혀있다.
4. $H$는 scalar multiplication에 대해 닫혀있다.



입니다. 여기서 첫 번째 조건이 어떤 vector space에 포함된 subset 조건을 나타내고, 두 번째부터 마지막 조건은 vector space을 만족하기 위한 조건을 나타냅니다. 



부분집합을 정의할 때 두 집합을 통해 정의하듯이, subspace가 정의되기 위해서는 두 vector space가 필요합니다.





<br/>

*example*



* Zero subspace



$V$에 존재하는 **zero vector**만을 가지는 집합은 $V$의 subspace가 됩니다. 이를 **zero subspace**라고 합니다.


$$
\{\boldsymbol{0}\} 
$$


다음 집합은 subspace의 조건을 만족합니다.



1. $\{\boldsymbol{0}\} \subseteq V$
2. $\boldsymbol{0} \in \{\boldsymbol{0}\}$
3. $\boldsymbol{u}, \boldsymbol{v} \in \{\boldsymbol{0}\} \Rightarrow \boldsymbol{u}= \boldsymbol{v}=\boldsymbol{0}, \boldsymbol{u}+ \boldsymbol{v}=\boldsymbol{0}$
4. $\boldsymbol{u}\in \{\boldsymbol{0}\}, k \in \mathbb R \Rightarrow k\boldsymbol{u}= 0 \in \{\boldsymbol{0}\}$





<br/>

*example*



The vector space $\mathbb R^2, \mathbb R^3$



$\mathbb{R}^2$와 $\mathbb R^3$는 다음과 같이 정의됩니다.


$$
\begin{aligned}

\mathbb{R}^2 &= \{\begin{bmatrix}x_1 \\ x_2 \end{bmatrix} \mid x_1, x_2 \in \mathbb R\}

\\ \\
\mathbb{R}^3 &= \{\begin{bmatrix}x_1 \\ x_2 \\ x_3\end{bmatrix} \mid x_1, x_2, x_3 \in \mathbb R\}

\end{aligned}
$$


두 집합 간 포함관계가 성립이 되지 않기 때문에, 두 vector space 간 subspace를 따질 수 없습니다.



<br/>

*example*



한편 다음 집합 $H$를 살펴봅시다.


$$
H= \{\begin{bmatrix}x_1 \\ x_2 \\ 0\end{bmatrix} \mid x_1, x_2 \in \mathbb R\}
$$


$H$의 경우 $\mathbb R^3$의 subspace가 됩니다.



1. $H \subseteq \mathbb R^3$
2.  $\boldsymbol{0} \in H$
3. $\boldsymbol{u}, \boldsymbol{v} \in H$이면


$$
\boldsymbol{u}=\begin{bmatrix}u_1 \\ u_2 \\ 0 \end{bmatrix}, \boldsymbol{v}=\begin{bmatrix}v_1 \\ v_2 \\ 0 \end{bmatrix}
$$


가 되어


$$
\boldsymbol{u+v} = \begin{bmatrix}u_1+v_1 \\ u_2+v_2 \\ 0 \end{bmatrix} \in H
$$


따라서 덧셈에 대해 닫혀 있습니다.



4. $\boldsymbol{u} \in H, k \in R$에 대해서

$$
\boldsymbol{u} =\begin{bmatrix}u_1 \\ u_2 \\ 0 \end{bmatrix}
$$



일 때


$$
k\boldsymbol{u} = \begin{bmatrix}ku_1 \\ ku_2 \\ 0 \end{bmatrix} \in H
$$


따라서 scalar multiplication에도 닫혀 있습니다.



subspace 조건을 모두 만족하기 때문에, $H$는 $\mathbb R^3$의 subspace입니다.





다음은 subspace와 span의 관계를 나타내는 정리에 대해 알아보겠습니다.



<br/>

**Theorem**



If $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}$ are in a vector space $V$, then



$$
Span\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}
$$


is a subspace of $V$





이 정리를 통해, the subset spanned by vectors in $V$($V$에 있는 vector로 spanned한 집합)은 subspace가 됩니다.



Given any subspace $H$ of $V$, a **spanning set for $H$** is a set $\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}$ in $H$ such that


$$
H=Span\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}
$$


$H$에 있는 특정 vector $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}$을 span하여 $H$를 만들었을 때, 이 vector들로 이루어진 집합을 **Spanning set for $H$**라고 합니다.



(증명은 appendix 참고)





<br/>

지금까지 vector space와 subspace에 대해 알아보았습니다. 다음 포스트에서는 linear transformation에 대해 알아보도록 하겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>

#### Appendix : Proof of Theorem



<br/>



**Theorem**



 



If $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}$ are in a vector space $V$, then



$$
Span\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\}
$$


is a subspace of $V$





* **Proof**


$$
H=Span\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\} = 
\{\boldsymbol{y} \mid \boldsymbol{y}=c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots+c_p\boldsymbol{v_p}, \ \ \ c_1, c_2, ..., c_p \in \mathbb R \}
$$


Span의 정의는 span을 구성하는 set에 속하는 vector들의 linear combination을 모두 모은 집합니다. 따라서, 이 집합이 subspace가 되기 위한 4가지 조건을 만족하는지 확인하면 됩니다.



1. $H \subseteq V$



$V$는 vector space이고, $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}$ 모두 $V$에 속한 vector이기 때문에, $\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}$의 linear combination 또한 $V$에 속합니다. 따라서 $Span\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_p}\} \subseteq V$을 만족합니다.



2. $\boldsymbol{0} \in H$



$c_1=c_2=...=c_p=0$인 경우, zero vector가 됩니다.



3. $\boldsymbol{u}, \boldsymbol{w} \in H$



$H$에 속하는 두 vector $\boldsymbol{u}, \boldsymbol{w}$을 다음과 같이 표현할 수 있습니다.


$$
\boldsymbol{u}=c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots+c_p\boldsymbol{v_p} \\
\boldsymbol{w}=d_1\boldsymbol{v_1}+d_2\boldsymbol{v_2}+\cdots+d_p\boldsymbol{v_p}
$$


두 vector를 더하면


$$
\begin{aligned}

\boldsymbol{u}+\boldsymbol{v}&=(c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots+c_p\boldsymbol{v_p})+(d_1\boldsymbol{v_1}+d_2\boldsymbol{v_2}+\cdots+d_p\boldsymbol{v_p}) \\
&=(c_1+d_1)\boldsymbol{v_1}+(c_2+d_2)\boldsymbol{v_2}+\cdots+(c_p+d_p)\boldsymbol{v_p} \in H

\end{aligned}
$$
가 되고 $H$에 속합니다. 따라서 $H$는 덧셈에 대해 닫혀있습니다.



4. $\boldsymbol{u} \in H, k \in \mathbb R$



$H$에 속하는 $\boldsymbol{u}$와 scalar $k$에 대해서


$$
\begin{aligned}

k\boldsymbol{u} &=k(c_1\boldsymbol{v_1}+c_2\boldsymbol{v_2}+\cdots+c_p\boldsymbol{v_p}) \\

&= kc_1\boldsymbol{v_1}+kc_2\boldsymbol{v_2}+\cdots+kc_p\boldsymbol{v_p} \in H

\end{aligned}
$$


가 되고 마찬가지로 $H$에 속합니다. 따라서 $H$는 scalar multiplication에 대해 닫혀있습니다.



$H$는 $V$의 subspace가 되기 위한 4가지 조건을 모두 만족하기 때문에, $H$는 $V$의 subspace입니다.