---
layout: single
title:  "이웃 기반 협업 필터링 (4)"
categories: [Recommender System]
tag: [Recommender System]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





<br/>

### 6. 이웃 방법론의 회귀 모델링 관점

<br/>

사용자 기반 및 아이템 기반 방법론 모두 이웃 사용자의 동일한 항목에 대한 평점 또는 이웃 항목의 동일한 사용자의 평점의 **선형 함수**로 평점을 예측한다. 사용자 기반 이웃 방법의 예측 함수를 살펴보자


$$
\hat r_{uj} = \mu_u + \frac{\sum_{v \in P_u(j)}Sim(u, v)(r_{vj}-\mu_v)}{\sum_{v\in P_u(j)}\begin{vmatrix}Sim(u, v)\end{vmatrix}}
$$




위의 식에서 예측 평점은 동일한 아이템의 다른 사용자의 평점의 linear combination이다. 이 때 linear combination 대상은 사용자 $u$와 충분히 유사한 취향을 가진 사용자에게 속하는 항목 $j$의 평점으로 제한되어 있다. 이는 $P_u(j)$를 통해 활성화된다. 만약 집합 $P_u(j)$를 아이템 $j$에 대한 모든 평점을 포함하는 집합으로 허용하면, 예측함수는 선형 회귀 모형과 유사해진다. Linear regression에서의 평점은 다른 평점의 가중 조합으로 예측되며 가중치(계수)는 optimization을 통해 결정된다. 이웃 기반 접근 방식에서는 선형 함수의 계수가 최적화 모델을 사용하는 대신 사용자-사용자 유사도와 추론 방식으로 선택된다.

아이템 기반 이웃 방법론의 경우에도 유사한 결과를 얻을 수 있다.


$$
\hat r_{ut} = \frac{\sum_{j \in Q_t(u)}AdjustedCosine(j, t)r_{uj}}{\sum_{j \in Q_t(u)}\begin{vmatrix} AdjustedCosine(j, t) \end{vmatrix}}
$$


집합 $Q_t(u)$는 사용자 $u$에 의해 평가된 대상 아이템 $t$의 최근접 아이템 집합을 나타낸다. 이 경우, 대상 항목 $t$에 대한 사용자 $u$의 평점은 자신의 평점의 선형 조합으로 표현된다. 사용자 기반 방법론의 경우와 마찬가지로 선형 조합의 계수는 유사도 값으로 휴리스틱하게 적용된다. 조합 가중치로 유사도 값을 사용하는 것은 휴리스틱적이고 임의적이다. 또한 계수는 아이템 간의 상호 종속성을 고려하지 않는다. 이러한 최근접 이웃 모델의 일반화 방법인 회귀 모델링 방법은 명확한 최적화 공식이 존재하고, 평점을 결합한 가중치가 더 합리적이라고 말할 수 있다. 

<br/>

#### 1) 사용자 기반 최근접 이웃 회귀

<br/>

항목 $j$에 대한 대상 사용자 $u$의 예측 평점을 모델링하기 위해 각 평점의 계수로 유사도가 아닌 새로운 parameter $w_{vu}^{user}$를 사용할 수 있다.


$$
\hat r_{uj} = \mu_u + \sum_{v \in P_u(j)}w_{vu}^{user}(r_{vj}-\mu_v)
$$


다만 $P_u(j)$를 형성하는 방법으로는 Pearson 상관계수나 유사도를 이용하여 정의한다. 다만 이전 $P_u(j)$ 형성 방법과의 차이점은 먼저 각 사용자에 대해 $k$명의 최근접 피어를 결정한 다음 평점이 관찰되는 피어만 유지해 정의된다. 따라서 $P_u(j)$ 집합의 크기는 $k$보다 훨씬 작은 경우가 많다.

직관적으로, 알지 못하는 계수 $w_{vu}^{user}$는 사용자 $u$에 의해 주어진 평점의 예측의 일부를 차지하고, 이는 사용자 $v$의 유사도에서 온다. 이 부분은 $w_{vu}^{user}(r_{vj}-\mu_v)$에 의해 주어졌기 때문이다. $w_{vu}^{user}$와 $w_{uv}^{user}$는 다를 수 있으며, $w_{vu}^{user}$는 Pearson 계수를 기준으로 사용자 $u$의 $k$명의 최근접 사용자 $v$에 의해서만 정의된다. 

위와 같이 문제를 설정하였다면, $w_{vu}^{user}$의 값을 추정하기 위해 예측된 평점 $\hat r_{uj}$와 관찰된 평점 $r_{uj}$ 사이의 제곱 차이를 이용할 수 있다. 즉, linear gregression 모델에서 최근접 $k$ 사용자와 사용자 $u$의 각 (관찰된) 평점을 예측한 다음, 오차를 계산하는 것이다.  최적화 문제는 사용자 $u$에 의해 결정된다. $I_u$를 타깃 사용자 $u$가 평가한 아이템의 집합이라고 할 때, loss function은 다음과 같이 정의할 수 있다.


$$
\begin{aligned}
J_u &= \sum_{j \in I_u}(r_{uj} -\hat{r}_{uj})^2 \\
&= \sum_{j \in I_u}\left(r_{uj} - \left[\mu_u + \sum_{v \in P_u(j)}w_{vu}^{user} (r_{vj}-\mu_v)\right]\right)^2
\end{aligned}
$$


모든 사용자 $u$를 통합한 objective function(cost function)은 다음과 같이 표시된다.


$$
\sum_{u=1}^m J_u = \sum_{u=1}^m \sum_{j \in I_u} \left(r_{uj} - \left[\mu_u + \sum_{v\in P_u(j)}w_{vu}^{user}(r_{vj}-\mu_v)\right]\right)^2
$$


각각의 사용자에 대해 최적화를 진행하는 것에 비해 전체 사용자에 대한 최적화를 진행하는 것이 계산량이 많지만, 모든 사용자를 다루기 때문에 다른 방법론과 쉽게 결합할 수 있다. 두 방법론 모두 least-square optimization problem이며, overfitting을 막기 위해 $w_{vu}^{user}$에 regularization term을 추가할수도 있다. regularization term을 추가한 모델은 다음과 같다.


$$
\sum_{u=1}^m J_u +\sum_{j \in I_u}\sum_{v \in P_u(j)}(w_{vu}^{user})^2 \\ =  \sum_{u=1}^m \sum_{j \in I_u} \left(r_{uj} - \left[\mu_u + \sum_{v\in P_u(j)}w_{vu}^{user}(r_{vj}-\mu_v)\right]\right)^2 +\sum_{j \in I_u}\sum_{v \in P_u(j)}(w_{vu}^{user})^2
$$

<br/>


#### (1) 희소성 및 bias 문제

<br/>

위 regression 접근 방식의 한 가지 문제는 $P_u(j)$의 크기가 동일한 사용자와 다양한 항목 index에 대해 다를 수 있다는 것이다. 이는 평점 행렬에 내재된 특별한 수준의 희소성 때문인데, 결과적으로 회귀 계수는 사용자 $u$와 함께 특정 아이템 $j$를 평가한 피어 사용자 수에 크게 의존하게 된다. 

ex : 대상 사용자 $u$가 <글래디에이터>와 <네로>를 모두 평가한 시나리오를 생각해보자. 대상 $u$의 $k$ 최근접 이웃 중 한 명의 사용자만 영화 <글래디에이터>를 평가하고, 모든 $k$명의 사용자는 <네로>를 평가했을 수 있다. 결과적으로 <글래디에이터>를 평가한 피어 사용자 $v$의 회귀 계수 $w_{vu}^{user}$는 <글래디에이터>를 평가한 유일한 사용자라는 사실에 크게 영향을 받는다. 이 경우 다른 평점 예측에 노이즈를 추가할 수 있기 때문에 overfitting이 발생할 수 있다.

이를 해결하기 위한 방법은 타깃 유저 $u$의 아이템 $j$의 평점을 예측할 때 $\frac{\begin{vmatrix}P_u(j)\end{vmatrix}}{k}$만큼의 비율만 예측하는 것이다. 회귀 계수가 대상 사용자의 모든 피어를 기반으로 하며 불완전한 정보는 분수로 보완해야한다. 그러므로, 이러한 접근은 회귀계수 식의 해석과 목적 함수 또한 변경시킨다. 이를 적용한 예측함수는 다음과 같다.


$$
\hat{r}_{uj} \cdot \frac{\begin{vmatrix}P_u(j)\end{vmatrix}}{k} = \mu_u + \sum_{v\in P_u(j)}w_{vu}^{user}(r_{vj}-\mu_v)
$$


또는 상수 부분인 $k$를 제거하고, $\mu_u$ 파트 역시 학습을 위해 bias 변수 $b_u$로 식을 변경할 수 있다.


$$
\hat r_{uj} = b_u^{user} + \frac{\sum_{v \in P_u(j)}w_{vu}^{user}(r_{vj}-b_v^{user})}{\sqrt{\begin{vmatrix}P_u(j)\end{vmatrix}}}
$$


또는 사용자 bias 뿐만 아니라 아이템 bias를 통합할 수 있다.


$$
\hat r_{uj} = b_{u}^{user} + b_{j}^{item} + \frac{\sum_{v \in P_u(j)}w_{vu}^{user}(r_{vj}-b_v^{user}-b_j^{item})}{\sqrt{\begin{vmatrix}P_u(j)\end{vmatrix}}}
$$




위 모델에서의 주요 문제는 계산복잡성이다. 계산 비용이 많이 들고, $m$ 사용자에 대한 $O(m^2)$ 공간이 필요하기에 모든 사용자-사용자 관계를 미리 계산하고 저장해야 한다. 이러한 모델은 아이템 공간이 빠르게 변경되는 설정에 적합하지만 사용자는 시간이 지남에 따라 상대적으로 안정적이다.



<br/>

#### 2) 아이템 기반 근접 이웃 회귀

<br/>

아이템 기반 접근 방식은 회귀가 사용자-사용자 상관관계가 아닌 아이템-아티메 상관관계를 학습하고 활용한다는 점을 제외하면 사용자 기반 접근 방식과 유사하다. 아이템 기반 방법론에서의 식을 생각해보자


$$
\hat r_{ut} = \frac{\sum_{j \in Q_t(u)}AdjustedCosine(j, t)r_{uj}}{\sum_{j \in Q_t(u)}\begin{vmatrix} AdjustedCosine(j, t) \end{vmatrix}}
$$


위 식 또한 $r_{uj}$에 대한 선형함수로 생각할 수 있으며 adjusted cosine 값이 아닌 우리가 추정해야 할 계수 $w_{jt}^{item}$으로 대체한 식으로 생각할 수 있다.


$$
\hat r_{ut} = \sum_{j \in Q_t(u)} w_{jt}^{item} r_{uj}
$$


$Q_t(u)$에서 최근접 아이템은 아이템 기반 이웃 방법에서와 같이 adjusted cosine을 사용해 결정할 수 있다. 사용자 기반 근접 이웃 회귀와 마찬가지로, 집합 $Q_t(u)$는 사용자 $u$가 평점을 제공한 대상 아이템 $t$의 $k$ 최근접 이웃의 하위 집합을 나타낸다. 

직관적으로, 알 수 없는 계수 $w_{jt}^{item}$은 $w_{jt}^{item} r_{uj}$에 의해 주어지기 때문에 아이템 $j$의 유사도에서 오는 평점 $t$의 평점의 일부를 차지한다. 마찬가지로, 실제 값과의 오차를 최소화시키는 방법을 통해 $w_{jt}^{item}$을 추정할 수 있다.

$U_t$가  대상 아이템 $t$를 평가한 사용자 집합일 때, 아이템 $t$에 대한 loss function은 다음과 같다.
$$
\begin{aligned}
J_t &= \sum_{u \in U_t}(r_{ut}-\hat r_{ut})^2 \\
&= \sum_{u \in U_t}\left(r_{ut} - \sum_{j \in Q_t(u)}w_{jt}^{item}r_{uj} \right)^2
\end{aligned}
$$


$t \in \{1, 2, ..., n\}$에 대해서 $w_{jt}$값들이 겹치지 않기 때문에, 모든 $t$를 고려한 cost function을 생각할 수 있다.


$$
\sum_{t=1}^nJ_t = \sum_{t=1}^n \sum_{u \in U_t} \left(r_{ut} - \sum_{j \in Q_t(u)} w_{jt}^{item}r_{uj}\right)^2
$$




$w_{jt}^{item}$에 regularization term을 추가하여 overfitting을 방지할 수 있다.




$$
\sum_{t=1}^nJ_t = \sum_{t=1}^n \sum_{u \in U_t} \left(r_{ut} - \sum_{j \in Q_t(u)} w_{jt}^{item}r_{uj}\right)^2 + \lambda \sum_{u \in U_t}\sum_{j \in Q_t(u)}(w_{jt}^{item})^2
$$




사용자 기반 근접 회귀 때와 마찬가지로 bias 변수를 추가하여 목적함수를 설정할 수 있다. 


$$
\hat r_{ut} = b_u^{user} + b_t^{item} + \frac{\sum_{j \in Q_t(u)}w_{jt}^{item}(r_{uj}-b_u^{user} - b_t^{item})}{\sqrt{\begin{vmatrix}Q_t(u)\end{vmatrix}}}
$$


이 경우 평점 행렬은 전체 평점 행렬의 평균으로부터 centered되어야 함을 가정한다. 따라서 전체 평균 중심으로 이동 후, 모델링을 진행하고, 마지막에 전체 평균 중심값을 다시 더해주어 모델링을 진행한다. 여기서 더 나아가, 괄호 안의 bias인 $b_u^{user} + b_t^{item}$은 통합된 상수 용어 $B_{uj}$로 대체할 수 있다. 이를 적용한 모델은 다음과 같다.


$$
\hat r_{ut} = b_u^{user} + b_t^{item} + \frac{\sum_{j \in Q_t(u)}w_{jt}^{item}(r_{uj}-B_{uj})}{\sqrt{\begin{vmatrix}Q_t(u)\end{vmatrix}}}
$$

<br/>


#### 3) 사용자 기반 및 아이템 기반 방법 결합

<br/>

사용자 기반 최근접 회귀 방법론과 아이템 기반 최근접 회귀 방법론을 결합하여 평점을 예측할 수 있다. 식은 다음과 같다. (마찬가지로 전체 평균으로 centering된 평점 행렬을 이용한다.)


$$
\hat r_{uj} = b_u^{user} + b_j^{item} + \frac{\sum_{v \in P_u(j)} w_{vu}^{user}(r_{vj} - B_{vj})}{\sqrt{\begin{vmatrix}P_u(j) \end{vmatrix}}} + \frac{\sum_{j \in Q_t(u)}w_{jt}^{item} (r_{uj}-B_{uj})}{\sqrt{\begin{vmatrix}Q_t(u)\end{vmatrix}}}
$$

<br/>


#### 4) 유사도 가중치를 이용하는 Joint Interpolation

<br/>

위 방법은 조인트 이웃 기반 방법을 다른 방법으로 접근한다. 먼저 사용자 기반 최근접 회귀 모델을 이용해 타깃 사용자 $u$의 평점을 예측한다. 그리고, 관측된 같은 아이템의 값과 비교하는 것 대신에, 타깃 사용자의 관측된 다른 아이템의 평점과 비교한다. 

$S$를 평점행렬에서의 관측된 모든 사용자-아이템 쌍의 집합이라고 하자.


$$
S = \{(u, t) : r_{ut} \ is \ observed \}
$$


이 후 objective function을 설정할 때, 비슷한 아이템일수록 objective function에서 가중치를 높게 부여하고, 다른 아이템일수록 objective function에서 가중치를 낮게 부여한다. 아이템 간의 유사도는 adjusted cosine을 사용하며, 예측된 값과 실제로 관측된 값  사이의 오차를 최소화하는 방법으로 진행된다. 이를 수식으로 나타내면 다음과 같다.


$$
\sum_{s : (u, s) \in S} \sum_{j: j\neq s} AdjustedCosine(j, s)(r_{us} - \hat r_{uj})^2\\
= \sum_{s:(u, s) \in S} \sum_{j : j \neq s} AdjustedCosine(j, s)\left(r_{us} - \left[\mu_u + \sum_{v \in P_u(j)}w_{vu}^{user}(r_{vj}-\mu_v) \right]\right)^2
$$




Regularization term을 추가하여 overfitting을 방지할 수 있다. 여기서 $P_u(j)$는 항목 $j$를 평가한 사용자 $u$를 타기팅하는 $k$ 최근접 사용자로 정의된다. 

목표 함수에서 각 개별 항의 곱셈 인자로 adjusted cosine을 사용함으로써, 유사한 항목의 대상 사용자의 평점을 더 유사하게 한다. 

1. 아이템-아이템 유사도는 예측된 평점이 유사한 항목의 관찰된 평점과 더 유사하도록 강제하기 위해 objective function에서의 항의 곱셈 인자로 사용된다.
2. 사용자-사용자 유사도는 타깃 사용자 $u$의 관련 피어 그룹 $P_u(j)$를 선택하는 용도로 사용된다. 

사용자와 아이템의 역할을 전환하는 것 또한 가능하지만 위 모델만큼 효과적이지는 않다.



<br/>

#### 5) 희소 선형 모델 (Sparse Linear Models)

<br/>

아이템 기반 근접 이웃 회귀를 기반으로 한 새로운 방법 중 하나이다. 해당 model family는 Sparse Linear Models이라고 하는데, regularization method를 사용하여 회귀 계수의 sparsity를 고려하기 때문이다. 이 방법은 음이 아닌 값에 대해서 적용이 가능하며, 따라서 평균 중심으로 centering하지 않은 평점 행렬을 기준으로 진행된다.  평균 중심 centering은 자동적으로 부정적인 평가를 제공하고, 이는 싫어함을 의미한다. 하지만 위 방법은 싫어함을 구체화할 수 있는 방법이 없는 경우에서도 작동하도록 설정되었다. 따라서 위 방법은 암시적 피드백 행렬에 적합하다. 암시적 피드백 행렬에서 관측되지 않은 값은 0으로 예측하는데, 최적화 모델은 이런한 값 중 일부를 매우 긍정적으로 예측할 수 있다. 따라서 이 접근 방식은 0으로 설정된 학습 항목에 대한 예측 오류를 기준으로 항목의 순위를 매기고 있다.

위 방법론은 회귀 계수를대상 아이템 $t$의 이웃으로만 제한하지 않는다. SLIM의 예측 함수는 다음과 같다.


$$
\hat r_{ut} = \sum_{j=1}^n w_{jt}^{item} r_{uj} \ \ u \in \{1, ..., m\}, \ \ t \in \{1, ..., n\}
$$




Overfitting을 제거하기 위해 우변의 타깃 아이템에 대한 평점 값은 포함하지 말아야 한다. 이는 $w_{tt}^{item} =0$이라는 제약 조건을 이용하여 해결할 수 있다. $\hat R = [\hat r_{uj}]$를 예측한 평점 행렬이고, $W^{item} = [w^{item}_{jt}]$를 아이템-아이템 행렬이라고 하자. 위의 제약조건을 적용하면 $W^{item}$의 diagonal entry는 모두 0이 되어야 한다. 위의 예측 함수를 행렬로 나타내면 다음과 가탇.


$$
\hat R = RW^{item} \\
Diagonal(W^{item}) = 0
$$




따라서, 위 모델에서의 목표는 $RW^{item}$과 $R$이 최대한 비슷해지는 $W^{item}$을 찾는 것이 된다. 행렬 간의 거리 지표 중 하나인 Frobenius normd을 이용하여 정의할 수 있으며 regularization term 또한 추가될 수 있다. 위 문제는 $W$의 column별로 분리가 가능하므로, 각각의 분리된 optimization 문제로 바라볼 수있다. 이를 적용한 타겟 아이템 t에 대한 objective function은 다음과 같이 정의된다.


$$
\begin{aligned}

J_t^s &= \sum_{u=1}^m(r_{ut}-\hat r_{ut})^2 + \lambda \sum_{j=1}^n(w_{jt}^{item})^2 + \lambda_1 \sum_{j=1}^n \begin{vmatrix} w_{jt}^{item}\end{vmatrix} \\
& = \sum_{u=1}^m(r_{ut} - \sum_{j=1}^n w_{jt}^{item} r_{uj})^2 + \lambda \sum_{j=1}^n(w_{jt}^{item})^2 + \lambda_1 \sum_{j=1}^n\begin{vmatrix}w_{jt}^{item}\end{vmatrix} \\
&subject \ \ to \\
&w_{jt}^{item} \geq 0 \ \ j \in \{1, ..., n\} \\
& w_{tt}^{item} = 0

\end{aligned}
$$


$J_t^s$의 마지막 두 term은 각각 $L_2, L_1$ regularization term으로, elastic-net regularization이라고 한다. 특히 $L_1$ term은 $w_{jt}^{item}$이 특정 임계값을 넘지 않으면 0으로 추정을 해주기 때문에, sparsity를 고려해주는 term으로 해석할 수 있다. Sparsity는 각각의 예측된 평점은 설명 가능한 적은 수의 관련된 아이템 평점의 linear combination으로 표현이 가능하다는 것을 보장해준다. 또한, 계수들이 모두 음이 아니기 때문에, 관련된 아이템은 모두 positive relation을 가진다. 위 문제는 coordinate descent method를 이용해서 풀 수 있다.

위 모델은 앞서 논의한 최근접 이웃 회귀 방법론과 비슷한데, 두 모델간 큰 차이점은 다음과 같다.



1. 최근접 이웃 회귀 방법론은 회귀 계수 제한을 타깃 사용자(또는 아이템)과 가장 비슷한, 가까운 $k$개로 제한을 한다. 하지만 SLIM 방법은 모든 아이템을 대상으로 추정을 하기 때문에, $k$보다 많은 수의 0이 아닌 계수값을 얻을 수있다. 또한 SLIM은 elastic-net을 이용하여 sparsity를 고려하지만, 최근접 이웃 회귀 방법은 최근접 이웃을 설정함으로써 sparsity를 고려하게 된다. 
2. SLIM 방법론은 우선적으로 암시적 피드백 데이터 세트를 위해 설계되었다. 위 방법은 평점이 긍정적 선호도만을 나타내는 경우에서도 사용이 가능하다. 이러한 방법론은 모델의 회귀 계수에 양수 설정을 가하는 방법론에 유용하다.
3. 최근접 이웃 회귀 방법론에서 회귀 계수는 양수 또는 음수일 수 있다. 하지만 SLIM의 계수는 음수가 아닌 계수로 제한된다. 
4. SLIM 방법론은 평점에 대한 예측 모델도 제안하지만, 예측 값의 최종적인 사용은 아이템의 순위를 매기는 것이다. 이 방법은 단항 행렬에서 자주 사용되기 때문에, 평점을 예측하는 것보다는 아이템 순위를 매기는 것으로 일반적으로 사용된다. 예측 값을 해석하는 다른 방법은 각 값을 평점 행렬에서 0이 아닌 평점을 0으로 대체하는 오류로 볼 수 있다. 오차가 클수록 평점의 예측값이 커지게 된다. 
5. SLIM 방법론은 추론 조정 계수와 함께 지정된 다양한 평점을 명시적으로 조정하지 않는다. 아이템의 존재가 유일한 정보인 단항 평점행렬의 경우, 조정 문제가 상대적으로 약하다. 따라서, 관측되지 않은 값을 0으로 바꾸는 것이 일반적이고, 좋고 싫음을 평점으로 나타낼 수 있는 경우와 비교해서 bias가 낮다.