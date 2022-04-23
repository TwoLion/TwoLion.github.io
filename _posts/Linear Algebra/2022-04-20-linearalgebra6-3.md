---
layout: single
title:  "6.3 Orthogonal Projection"
categories: [Linear Algebra]
tag: [Linear Algebra, orthogonal projection]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---



이번 포스트에서는 orthogonal projection에 대해 알아보겠습니다.



<br/>



### 1) Orthogonal Projection



<br/>

#### (1) Purpose of orthogonal projection

<br/>





Orthogonal projection의 목표는 다음과 같습니다. 



$\mathbb R^n$의 subspace $W$와 벡터 $\boldsymbol y$에 대해서 **$W$에 속하는 벡터** 중 다음 조건을 만족하는 벡터 $\hat {\boldsymbol y}$을 찾고 싶습니다. 

1. $\boldsymbol{y}-\hat{\boldsymbol{y}}$가 $W$에 orthogonal합니다. 


$$
(\boldsymbol{y} - \hat{\boldsymbol y}) \perp W
$$


2. $\boldsymbol y$에서 $W$에 가장 가까운 벡터가 $\hat{\boldsymbol{y}}$입니다.


$$
\|\boldsymbol y -\hat{\boldsymbol y}\| \leq \|\boldsymbol{y}-\boldsymbol x\| \ \ for \ \ all \ \ x\in W
$$


해당 상황을 시각적으로 표현하면 다음과 같습니다. 



![선대 projection](../../images/2022-04-20-linearalgebra6-3/선대 projection.png)





이러한 $\hat{\boldsymbol{y}}$는 unique하게 존재하는데, 왜 unique하게 존재하는지, 어떻게 해당 벡터를 찾을 수 있는지 알아보겠습니다. 



<br/>



$\{\boldsymbol{u_1}, ..., \boldsymbol{u_n}\}$이 $\mathbb R^n$의 orthogonal basis면, $\boldsymbol y \in \mathbb R^n$은 


$$
\boldsymbol{y} = \boldsymbol{z_1} + \boldsymbol{z_2}
$$


로 표현이 가능합니다. 여기서, $\boldsymbol{z_1}$은 basis에 속한 몇 개의 벡터의 linear combination으로 표현되는 벡터이고, $\boldsymbol{z_2}$는 basis에 속한 벡터 중 $\boldsymbol{z_1}$에 포함된 벡터를 제외한 나머지 벡터들의 linear combination으로 표현되는 벡터입니다.

**이 때, $\boldsymbol{z_1}$과 $\boldsymbol{z_2}$는 orthogonal합니다.** 예를 들어


$$
\boldsymbol{z_1} = c_1\boldsymbol{u_1}+\cdots+c_k\boldsymbol{u_k}\\
\boldsymbol{z_2} = c_{k+1}\boldsymbol{u_{k+1}} + \cdots + c_n\boldsymbol{u_n}
$$


일 경우, 


$$
\boldsymbol{z_1} \perp \boldsymbol{z_2}
$$


 가 됩니다. 또한 $\boldsymbol{z_1}$을 표현할 때 사용한 벡터 $\{\boldsymbol{u_1}, ..., \boldsymbol{u_k}\}$를 basis로 하는 subspace $W$를 생각한다면, $\boldsymbol{z_1}\in W$을 만족합니다. 이 때, $\{\boldsymbol{u_1}, ..., \boldsymbol{u_n}\}$이 orthogonal basis이므로,


$$
\boldsymbol{z_2} \in W^\perp
$$
 

를 만족합니다. 또한, $\boldsymbol{z_2}$을 표현할 때 사용한 벡터 $\{\boldsymbol{u_{k+1}, ..., \boldsymbol{u_{n}}}\}$을 basis로 하는 subspace는 바로 $W^\perp$입니다.



이 성질을 이용하면 orthogonal projection의 목적에 부합하는 $\hat{\boldsymbol y}$를 찾을 수 있습니다.



<br/>

**Theorem**



Let $W$ be a subspace of $\mathbb R^n$. Then each $\boldsymbol{y}$ in $\mathbb R^n$ can be written uniquely in the form


$$
\boldsymbol{y} =\hat{\boldsymbol{y}} +\boldsymbol{z}
$$


where $\hat{\boldsymbol{y}} \in W$ and $\boldsymbol z \in W^\perp$. 

In fact, if $\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$ is any orthogonal basis for $W$, then 


$$
\hat{\boldsymbol y} = \frac{\boldsymbol{u_1}\cdot \boldsymbol{y}}{\boldsymbol{u_1}\cdot \boldsymbol{u_1}}\boldsymbol{u_1} +
\frac{\boldsymbol{u_2}\cdot \boldsymbol{y}}{\boldsymbol{u_2}\cdot \boldsymbol{u_2}}\boldsymbol{u_2} + \cdots +
\frac{\boldsymbol{u_p}\cdot \boldsymbol{y}}{\boldsymbol{u_p}\cdot \boldsymbol{u_p}}\boldsymbol{u_p}
$$




