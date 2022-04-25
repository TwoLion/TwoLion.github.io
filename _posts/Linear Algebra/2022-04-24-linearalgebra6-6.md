---
layout: single
title:  "6.6 Inner Product Spaces"
categories: [Linear Algebra]
tag: [Linear Algebra, inner product space]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---







이번 포스트에서는 inner product space에 대해 알아보겠습니다.



<br/>

### 1) Inner product space



<br/>

#### (1) Inner product space

<br/>



Vector space에서 addition과 scalar multiple 못지 않게 중요한 연산 중 하나가 inner product입니다. 이전 포스트에서 사용한 $\mathbb R^n$에서의 inner product


$$
\boldsymbol{x}\cdot\boldsymbol{y} = \boldsymbol{x}^T\boldsymbol{y}
$$


를 이용하여 length(norm), distance, orthogonality를 정의할 수 있었습니다. 하지만 $\mathbb R^n$이 아닌 vector space 또한 존재하고, 이러한 vector space에서는 위의 inner product가 성립하지 않을 수 있습니다. 따라서, vector space에서 특정한 조건을 만족하는 연산을 inner product라고 따로 정의하게 됩니다. 그렇다면 일반적인 vector space에서의 inner product를 어떻게 정의하는지 알아보겠습니다.



<br/>

**Definition : Inner product**



An inner product on a vector space $V$ is a function that, to each pair of vectors $\boldsymbol{u}, \boldsymbol{v}$ in $V$, associates a real number


$$
\langle\boldsymbol{u}, \boldsymbol{v}\rangle
$$


and satisfies the following axioms, for all $\boldsymbol{u}, \boldsymbol{v}, \boldsymbol{w}$ in $V$ and all scalars $c$,

1. $\langle\boldsymbol{u}, \boldsymbol{v}\rangle \ =\  \langle\boldsymbol{v}, \boldsymbol{u}\rangle$
2. $\langle\boldsymbol{u+v}, \boldsymbol{w}\rangle =\langle\boldsymbol{u}, \boldsymbol{w}\rangle + \langle\boldsymbol{v}, \boldsymbol{w}\rangle$
3. $\langle c\boldsymbol{u}, \boldsymbol{v}\rangle = c\langle \boldsymbol{u}, \boldsymbol{v}\rangle $
4. $\langle\boldsymbol{u}, \boldsymbol{u} \rangle\geq 0$ and $\langle \boldsymbol{u}, \boldsymbol{u}\rangle=0$ if and only if $\boldsymbol{u}=0$



A vector space with an inner product is called an inner prodcut space



**Vector space $V$의 임의의 두 벡터를 input으로 넣었을 때 output이 실수가 되는 함수 중 다음 4가지 조건을 만족하는 함수를 해당 vector space에서의 inner prodcut라고 합니다. 또한 inner product 연산이 정의된 vector space를 inner product space라고 합니다.**



<br/>

*example*



$\mathbb R^2$에서의 벡터 $\boldsymbol{u} = (u_1, u_2), \  \boldsymbol{v} = (v_1, v_2), \ \boldsymbol{w}=(w_1, w_2)$에 대해 inner product를 


$$
\langle\boldsymbol{u}, \boldsymbol{v} \rangle = a_1u_1v_1 + a_2u_2v_2, \ \ \ a_1, a_2>0
$$


이라고 정의하면, 해당 연산은 inner product 성질을 만족합니다. 해당 연산이 inner product가 되려면 4가지 조건을 만족하는지 확인해보면 됩니다. 



첫 번째 조건을 만족하는지 확인해보면


$$
\langle \boldsymbol{u}, \boldsymbol{v}\rangle = a_1u_1v_1+a_2u_2v_2 \\
\langle \boldsymbol{v}, \boldsymbol{u}\rangle = a_1v_1u_1+a_2v_2u_2 \\
\langle\boldsymbol{u}, \boldsymbol{v}\rangle = \langle \boldsymbol{v}, \boldsymbol{u} \rangle
$$


임을 알 수 있습니다.

두 번째 조건을 화인하면
$$
\langle \boldsymbol{u}+\boldsymbol{v}, \boldsymbol{w}\rangle = a_1(u_1+v_1)w_1 + a_2(u_2+v_2)w_2 \\
\langle \boldsymbol{u}, \boldsymbol{w}\rangle + \langle \boldsymbol{v}, \boldsymbol{w}\rangle = a_1u_1w_1 + a_2u_2w_2 + a_1v_1w_1 + a_2v_2w_2 = a_1(u_1+v_1)w_1 + a_2(u_2+v_2)w_2 \\
\langle \boldsymbol{u}+\boldsymbol{v},\boldsymbol{w}\rangle = \langle \boldsymbol{u}, \boldsymbol{w}\rangle + \langle\boldsymbol{v}, \boldsymbol{w}\rangle
$$


이므로 두 번째 조건도 만족합니다.



세 번째로, 


$$
\langle c\boldsymbol{u}, \boldsymbol{v} \rangle = a_1cu_1v_1 + a_2cu_2v_2 = c(a_1u_1v_1+a_2u_2v_2)=c\langle \boldsymbol{u}, \boldsymbol{v} \rangle
$$


을 만족합니다.



마지막으로


$$
\langle\boldsymbol{u}, \boldsymbol{u}\rangle = a_1u_1^2 + a_2u_2^2 \geq 0
$$


을 만족합니다. ($a_1, a_2>0$) 또한 $\langle \boldsymbol{u}, \boldsymbol{u}\rangle=0$ 이 되려면 $\boldsymbol{u}=0$일 때만 성립합니다.

따라서, 해당 연산은 inner product 조건을 만족합니다. 



<br/>

*example*



Let $\mathbb P_n$ denote


$$
\mathbb P_n = \{p(t) \mid p(t) = a_0+a_1t+\cdots + a_nt^n, \ \ a_1, ..., a_n\in\mathbb R\}
$$


즉 order(degree, 차수)가 $n$보다 작거나 같은 모든 polynomial의 집합이 $\mathbb P_n$입니다. 먼저, 해당 set은 vector space입니다. 이는



1. $a_0=\cdots=a_n=0$으로 설정하면, zero polynomial이 $\mathbb P_n$에 속하기 때문입니다. 해당 set에서 zero vector의 역할을 하는 polynomial이 zero polynomial입니다.
2. $p(t), q(t) \in \mathbb P_n$이면, $p(t), q(t)$ 모두 차수가 n보다 작거나 같은 polynomial입니다. 따라서 이 두 polynomial의 합 또한 차수가 $n$보다 낮거나 같고, 계수가 모두 실수인 polynomial이므로, $p(t)+q(t) \in\mathbb P_n$을 만족합니다.
3. $p(t)\in \mathbb P_n$, scalar $c\in \mathbb R$에 대해서 $cp(t)$의 차수가 $n$보다 작거나 같고, 계수가 모두 실수이므로 $cp(t)\in\mathbb P_n$을 만족합니다.



따라서 $\mathbb P_n$은 vector space입니다. 



다음으로 inner product 연산을 다음과 같이 정의합니다.



Let $t_0, ..., t_n$ be distinct real numbers. For $p$ and $q$ in $\mathbb P_n$, define


$$
\langle p, q \rangle = p(t_0)q(t_0)+p(t_1)q(t_1)+\cdots+p(t_n)q(t_n)
$$


이 때 해당 연산은 inner product 조건을 만족합니다. 

먼저, 


$$
\langle p, q\rangle = p(t_0)q(t_0)+\cdots+p(t_n)q(t_n) = q(t_0)p(t_0)+\cdots+q(t_n)p(t_n)=\langle q, p \rangle
$$


을 만족합니다. 두 번째로 $r\in P_n$에 대해서


$$
\begin{aligned}

\langle p+q, r\rangle &= (p(t_0)+q(t_0))r(t_0)+\cdots(p(t_n)+q(t_q))r(t_n) \\
&=(p(t_0)r(t_0)+\cdots p(t_n)r(t_n))+(q(t_0)r(t_0)+\cdots + q(t_n)r(t_n)) \\
&=\langle p, r \rangle + \langle q, r \rangle

\end{aligned}
$$


를 만족합니다. 세 번째로 $c \in \mathbb R$에 대해


$$
\langle cp, q \rangle = cp(t_0)q(t_0)+\cdots + cp(t_n)q(t_n) = c(p(t_0)q(t_0)+\cdots p(t_n)q(t_n)) = c\langle p, q \rangle
$$


를 만족합니다. 마지막으로


$$
\langle p, p \rangle = \{p(t_0)\}^2 +\cdots \{p(t_n)\}^2 \geq 0
$$


을 만족하고, 


$$
\{p(t_0)\}^2 +\cdots \{p(t_n)\}^2 =0
$$


을 만족하려면


$$
p(t_0) = \cdots = p(t_n) =0
$$


가 성립해야 합니다. 현재 $p(t)$는 order가 $n$인데 $p(t)=0$의 solution이 $n+1$개여야 하므로, 이를 만족시키는 $p(t)$는 zero polynomial


$$
p(t)=0
$$


밖에 없습니다. 따라서 inner product가 되기 위한 4개의 조건을 모두 만족하므로 해당 연산은 $\mathbb P_n$의 inner product로 정의할 수 있습니다.



<br/>

#### (2) Lengths, Distances, Orthogonality

<br/>



임의의 vector space에서 inner product를 정의를 하였으니, length(norm), distance, orthogonality를 정의할 수 있습니다. 



<br/>

**Definition**



Let $V$ is vector space with inner product $\langle\boldsymbol{u}, \boldsymbol{v}\rangle$($\boldsymbol{u}, \boldsymbol{v}\in V$) the length(norm) of $\boldsymbol{u}$ is


$$
\|\boldsymbol{u}\| = \sqrt{\langle \boldsymbol{u}, \boldsymbol{u}\rangle}
$$


The distance between $\boldsymbol{u}$ and $\boldsymbol{v}$ is


$$
dist(\boldsymbol{u}, \boldsymbol{v}) = \|\boldsymbol{u}-\boldsymbol{v}\|
$$


The vector $\boldsymbol{u}, \boldsymbol{v}$ is orthogonal if


$$
\langle \boldsymbol{u}, \boldsymbol{v} \rangle = 0
$$


다음과 같이 length, distance, orthogonality를 정의하였기 때문에, 이전 포스트에서 배운 orthogonal set, orthogonal basis, orthogonal projection, Gram-Schmidt process, best approximation 모두 똑같은 방법으로 정의할 수 있습니다. 달라지는 것은 vector space와 해당 vector space에서의 inner product입니다.





<br/>



#### (3) Two Inequalities



<br/>



Inner product space에서 성립하는 두 가지의 부등식이 있습니다. Cauchy-Schwarz inequality와 triangle inequality를 알아보도록 하겠습니다. 



<br/>

**Theorem : The Cauchy-Schwarz Inequality**



For all $\boldsymbol{u}, \boldsymbol{v}$ in $V$


$$
\begin{vmatrix}\langle\boldsymbol{u}, \boldsymbol{v} \rangle\end{vmatrix} \leq \|\boldsymbol{u}\|\|\boldsymbol{v}\|
$$




두 벡터의 inner product값은 두 벡터의 length의 곱보다 작거나 같습니다.



<br/>

**Theorem : The Triangle Inequality**



For all $\boldsymbol{u}, \boldsymbol{v}$ in $V$


$$
\|\boldsymbol{u}+\boldsymbol{v}\| \leq \|\boldsymbol{u}\| + \|\boldsymbol{v}\|
$$


두 벡터의 합의 length는 각각의 벡터의 length의 합보다 작거나 같습니다.





<br/>



지금까지 inner product space에 대해 알아보았습니다. 다음 포스트에서는 orthogonal projection의 응용인 linear regression과 weighted least squares에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>

### Appendix : Proof of theorem



<br/>

**Theorem : The Cauchy-Schwarz Inequality**



For all $\boldsymbol{u}, \boldsymbol{v}$ in $V$


$$
\begin{vmatrix}\langle\boldsymbol{u}, \boldsymbol{v} \rangle\end{vmatrix} \leq \|\boldsymbol{u}\|\|\boldsymbol{v}\|
$$




<br/>

* **Proof**



scalar $c$에 대해서, ($c \in \mathbb R$)


$$
\|\boldsymbol{u}+c\boldsymbol{v}\|^2=\langle \boldsymbol{u}+c\boldsymbol{v}, \boldsymbol{u}+c\boldsymbol{v}\rangle = \|\boldsymbol{u}\|^2 + 2c\langle\boldsymbol{u}, \boldsymbol{v} \rangle + c^2\|\boldsymbol{v}\| \geq 0
$$


을 만족합니다. 이는 임의의 scalar $c$에 대해서 성립하기 때문에, 위 식을 $c$에 대한 2차식으로 생각하면, 판별식


$$
(\langle \boldsymbol{u}, \boldsymbol{v}  \rangle)^2-\|\boldsymbol{u}\|^2\|\boldsymbol{v}\|^2 \leq 0
$$


 를 만족해야 합니다. 이를 정리하면


$$
(\langle \boldsymbol{u}, \boldsymbol{v}  \rangle)^2 \leq \|\boldsymbol{u}\|^2\|\boldsymbol{v}\|^2 = (\|\boldsymbol u \| \|\boldsymbol v \|)^2
$$


가 되어


$$
\begin{vmatrix}\langle\boldsymbol{u}, \boldsymbol{v} \rangle\end{vmatrix} \leq \|\boldsymbol{u}\|\|\boldsymbol{v}\|
$$


를 만족합니다. 



 <br/>



* Another proof

$\boldsymbol{u}=0$인 경우, 양변이 모두 $0$이므로 부등식이 성립합니다. $\boldsymbol{u}\neq 0$ 경우 다음의 subspace $W$


$$
W =Span\{\boldsymbol{u}\}
$$


를 생각해봅시다. 이 때,


$$
proj_{W}\boldsymbol{v} = \frac{\langle\boldsymbol{u}, \boldsymbol{v} \rangle}{\langle\boldsymbol{u}, \boldsymbol{u} \rangle}\boldsymbol{u}
$$


인 것을 알 수 있습니다. 해당 projection vector의 length를 구해보면


$$
\|proj_W\boldsymbol{v}\| = \left|\left|\frac{\langle\boldsymbol{u}, \boldsymbol{v} \rangle}{\langle \boldsymbol{u}, \boldsymbol{u} \rangle}\boldsymbol{u}\right|\right| = \left|\frac{\langle\boldsymbol{u}, \boldsymbol{v} \rangle}{\langle \boldsymbol{u}, \boldsymbol{u} \rangle}\right|\|\boldsymbol{u}\| = \frac{\left|\langle \boldsymbol{u}, \boldsymbol{v}\rangle\right|}{\|\boldsymbol{u}\|}
$$


입니다. 


$$
\|proj_W\boldsymbol{v}\| \leq \|\boldsymbol v\|
$$


이므로


$$
\begin{vmatrix}\langle\boldsymbol{u}, \boldsymbol{v} \rangle\end{vmatrix} \leq \|\boldsymbol{u}\|\|\boldsymbol{v}\|
$$




가 성립합니다.





<br/>

**Theorem : The Triangle Inequality**



For all $\boldsymbol{u}, \boldsymbol{v}$ in $V$


$$
\|\boldsymbol{u}+\boldsymbol{v}\| \leq \|\boldsymbol{u}\| + \|\boldsymbol{v}\|
$$


<br/>



* **Proof**


$$
\|\boldsymbol{u}+\boldsymbol{v}\|^2 = \|\boldsymbol{u}\|^2 + 2\langle \boldsymbol{u}, \boldsymbol{v} \rangle + \|\boldsymbol{v}\|^2
$$


입니다. 이 때 Cauchy-Schartz inequality에 의해


$$
\|\boldsymbol{u}\|^2 + 2\langle \boldsymbol{u}, \boldsymbol{v} \rangle + \|\boldsymbol{v}\|^2 \leq 
\|\boldsymbol{u}\|^2 + 2\|\boldsymbol{u}\|\|\boldsymbol{v}\|+\|\boldsymbol{v}\|^2 = (\|\boldsymbol{u}\|+\|\boldsymbol{v}\|)^2
$$
 

을 만족하므로


$$
\|\boldsymbol{u}+\boldsymbol{v}\| \leq \|\boldsymbol{u}\|+\|\boldsymbol{v}\|
$$




를 얻을 수 있습니다.





