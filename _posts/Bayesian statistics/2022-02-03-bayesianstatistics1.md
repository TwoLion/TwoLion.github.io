---
layout: single
title: "1.1 Introduction to Bayesian statistics (1)"
categories: [Bayesian Statistics]
tag: [Bayesian Statistics]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---













이번 포스트에서는 베이지안 통계의 기본적인 framework와 frequentist와 bayesian의 차이에 대해 전반적으로 알아보겠습니다.



<br/>





### 1. Notation



<br/>



#### 1) Notation

<br/>



기본적으로, 통계적 추론에서 사용되는 용어의 notation은 다음과 같습니다.



* $\theta$ : parameter of interest(unknown) : 모수이며, 알려지지 않은 값입니다.

* $X$ : random variable : 확률변수

* $x$ : A realization value of $X$. The observation(data) : 관측값

* Statistical model : The distribution for $X$ given the parameter $\theta$ 

  





통계학에서의 목표 중 하나는 data, 즉 $x$를 이용하여 적절한 parameter $\theta$를 추정하는 것입니다.





* Likelihood



pdf of $X$ given $\theta$를 $\theta$에 대한 함수로 바라보았을 때, likelihood function이 됩니다.


$$
L(\theta | x) = f(x |\theta)
$$


$\theta$는 unknown이므로, data $x$를 통하여 $\theta$를 추정해야 합니다. 어떤 parameter 값  $\theta_1$, $\theta_2$에 대해서


$$
f(x|\theta_1) < f(x|\theta_2)
$$


를 만족한다면, observation $x$는 $f(x \mid \theta_2)$에서 추출되었다고 생각하는 것이 합리적입니다. $\theta_2$일 때 $x$가 관측될 확률이 더 크기 때문입니다.



따라서, likelihood를 최대로 만들어주는 estimator $\theta$를 Maximum Likelihhod Estimator라고 합니다. 







<br/>



### 2. Frequentist vs Bayesian

<br/>



위의 notation에 대해서 Bayesian과 Frequentist의 접근 방법은 차이가 있습니다. 



<br/>



#### 1) Frequentist approach



<br/>

* A frequentist procedure quantifies uncertainty in terms of repeating the process that generated the data many times



Frequentist는 불확실성을 데이터를 반복하여 생성하는 과정에서 논합니다. 데이터를 생성할 때마다 각 **데이터마다의 오차 또는 차이에 의해 발생하는 불확실성을 고려합니다.**



* Parameter $\theta$는 **fix되어 있고**, 모르는 값입니다. 
* Sample data $X$는 random입니다. 즉 $X$는 random variable(or vector)입니다. 



$\theta$가 fix되어 있기 때문에,


$$
P(\theta>0.2)=0.1
$$


와 같은 식은 성립하지 않습니다. $\theta$는 우리가 모르는 값이지만 고정이 되어 있기 때문에, $\theta$가 0.2보다 크면 1, 0.2보다 작거나 같으면 0을 가집니다. 정리하면 $\theta$는 random variable이 아니므로, 위의 식은 성립하지 않습니다.

$\theta$가 random variable이 아니기 때문에, 신뢰구간 해석, 가설 검정 결과를 해석할 때 주의하여야 합니다.

먼저,  $\theta$의 distribution을 정의하지 않습니다. 

 $\theta$가 95% 신뢰구간에 포함된 확률이 95%라고 해석하는 것은 잘못된 해석입니다.

마지막으로, 귀무가설이 기각될 확률, 대립가설이 채택될 확률 자체를 정의하지 않습니다.





* 확률과 관련된 statement는 parameter $\theta$가 아닌 randomness of data에 의해 발생합니다.

* Statistic $\hat\theta$는 Sample의 summary입니다.

  예를 들어 sample mean $\bar{X}$는 statistic이고, 이는 population mean $\mu$의 estimator입니다. 

* $\hat{\theta}$는 sample의 summary, 즉 random variable의 함수이므로, $\hat\theta$는 distribution이 존재합니다. 위 distribution은 data를 반복적으로 추출하였을 때 각 시행마다 발생하는 $\hat\theta$의 분포로, 이를 **sampling distribution**이라고 합니다.

* 95% confidence interval (l, u)는 실제 $\theta$에 대해 반복적으로 data를 추출하고 신뢰구간을 만들면,만들어진 신뢰구간 중 95%가 $\theta$를 포함하고 있다는 것을 말합니다.

* 가설 검정에서 사용되는 p-value는 귀무가설 $H_0$가 참일 때, test statistic이 극단적인 값을 가질 ($H_1$을 지지하는 쪽으로) 확률을 뜻합니다. 



즉, 가설검정이나, 신뢰구간을 설명할 때 $\theta$가 아닌 data로부터 얻은 statistic $\hat\theta$를 이용하여 설명합니다.



<br/>



*example*



앞이 나올 확률이  $\theta$를 100번 던져 60번 앞이 나왔다고 가정해봅시다. $X \sim Bin(100, \theta)$인 $X$에 대해 $X$의 관측값 $x=60$인 상황입니다.



* Frequentist 입장에서, maximum likelihood method를 사용해서 얻은 estimates $\hat\theta$는 $0.6$입니다.
* $P(\theta>0.5)$는 따로 정의하지 않습니다. 



<br/>



#### 2) Bayesian approach



<br/>



* Bayesians view $\theta$ as fixed and unknown



Bayesian 역시 $\theta$를 fixed와 unknown으로 생각합니다. 



* However, we express our uncertainty about $\theta$ **using probability distribution**



하지만, $\theta$에 대한 불확실성을 나타내기 위해, $\theta$를 **random variable**로 나타내고, 따라서 $\theta$에 **distribution**이 존재합니다.



* Observation을 관찰하기 전 $\theta$의 분포를 **prior distribution**이라고 합니다.
* prior distribution은 분석에 따라 자유롭게 선택할 수 있습니다. 즉, prior 선택에 있어 주관이 들어갈 수 있습니다.





* Our uncertainty about $\theta$ is changed(hopefully reduced) after observing the data.



Bayesian에서는 parameter $\theta$에 대한 불확실성을 직접 논합니다. 이 불확실성은 data를 관측하고 난 뒤 변화합니다.



* Likelihood function is the distribution of the observed data given the parameters



Likelihood 또한 보는 관점이 다릅니다. parameter가 주어졌을 때, observed data의 distribution의 관점에서  해석합니다.

Likelihood function은 frequentist 입장에서 해석할 때 사용한 likelihood function과 동일합니다. 따라서, 만약 prior distribution의 영향이 적다면, Bayesian과 maximum likelihood analysis의 결과는 비슷합니다. 



* data를 관측한 후의 $\theta$의 분포를 **Posterior distribution**이라고 합니다. 



* **Bayes theorem**에 의해, prior distribution과 likelihood function을 이용하여 posterior distribution을 정의할 수 있습니다.


$$
\pi(\theta | X) = \frac{f(x|\theta)\pi(\theta)}{f(x)}
$$


where


$$
f(x) = \int f(x|\theta)\pi(\theta)d\theta
$$


즉,



**Posterior $\propto$ prior $\times$ likelihood**



을 만족합니다. 



* Bayesian은 $\theta$에 대해 해석할 때, 우리가 관측한 $x$만을 이용하여(given되었을 때) 해석합니다. 하지만 frequentist의 경우, 가상으로 반복 생성된 자료에 기반하여 $\theta$를 해석합니다. (ex: 신뢰구간, 가설 검정)



<br/>



*example*



앞이 나올 확률이  $\theta$를 100번 던져 60번 앞이 나왔다고 가정해봅시다. $X \sim Bin(100, \theta)$인 $X$에 대해 $X$의 관측값 $x=60$인 상황입니다. (앞의 예제와 같습니다.)



Bayesian에서는 $\theta$를 random variable로 생각합니다. $\theta \in [0, 1]$입니다. 

예를 들어, $\theta$에 대한 prior를 $Unif(0, 1)$로 두었을 때


$$
\pi(\theta) = I(0<\theta<1), \ f(x | \theta) = {100\choose x}\theta^{x}(1-\theta)^{100-x}
$$
이므로, 이를 이용하여 posterior distribution을 구하면


$$
\pi(\theta | X=60) \propto \theta^{60}(1-\theta)^{40}
$$


이고, 이는 $Beta(61, 41)$의 kernel이므로, posterior distribution은 $Beta(61, 41)$이 됩니다. 



prior를 uniform distribution이 아닌 $Beta(\alpha, \beta)$ distribution으로도 설정할 수 있습니다. 이 경우 posterior는


$$
\pi(\theta | X=x) \propto \theta^{x+\alpha-1}(1-\theta)^{100-x+\beta-1}
$$


가 되어, $Beta(x+\alpha, 100-x+\beta)$ distribution이 됩니다.



따라서, $\alpha, \beta$ 값에 따라 posterior distribution이 달라지게 됩니다. 만약 예전 실험 정보에서, $\theta \approx 0.8$이라는 결과를 알고 있었다면, prior distribution의 mean이 0.8이 나오도록 $\alpha, \beta$를 정해줄 수 있습니다.





<br/>

#### 3) Meaning of probability

<br/>



확률의 의미 또한 frequentist와 bayeisan 간 입장 차이가 있습니다.



* Frequentist 

  Frequentist 입장에서 확률은 실험의 무한한 반복의 의미를 포함합니다. 여기서 사건 $A$가 발생할 확률을 다음과 같이 정의합니다.

  
  $$
  P(A) = \frac{frequency \ of \ A}{the \ number \ of \ repetitions}
  $$



* Bayesian

  Bayesian 입장에서 확률은 개인적인 믿음, 또는 불확실성의 측도(measure)입니다. 따라서 불확실성을 probability로 표현할 수 있다고 생각합니다. (parameter $\theta$를 random variable로 해석하는 것처럼 말이죠.)





지금까지 Bayesian과 Frequentist의 차이에 대해서 알아보았습니다. 다음 포스트에서는 베이지안 통계에서의 추론 과정에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!

