---
layout: single
title:  "Kernel Smoothing Method"
categories: [Statistical Learning]
tag: [Kernel Smoothing Method, Local Regression]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---









Local Regression 정리



<br/>



### 1) Kernel Smoothing Method



<br/>



#### Idea



<br/>




$$
y=f(x)
$$


를 추정할 때, parametric한 model을 가정하지 않고, point마다 point 근방의 데이터를 이용하여 $y$값을 추정할 수 있음



*example* 



 KNN method가 근방의 데이터를 이용하여 outcome을 estimate한 예시, KNN method에서


$$
E(Y | X=x) = \hat{f}(x)=Ave(y_i | x_i \in N_k(x))
$$




where


$$
N_k(x)\ : \ The \ set \ of \ k \ points \ nearest \ to \ x \ in \ squared \ distance
$$




Problem of KNN



$x$값이 바뀜에 따라 바뀐 $x$값과 가까운 $k$개의 observation 또한 변화함 - $\hat{f}(x)$ 그래프가 연속적이지 않음 - ugly and unnecessary.



Idea : KNN과 같이 근방의 데이터를 활용하되, $\hat{f}(x)$가 smooth하도록 setting할 수 없을까?



Solution : Kernel weighting : $x$와의 거리에 따라 weight의 변화를 주어 outcome estimate하기



<br/>



#### Nadaraya-watson kernel-weighted average



<br/>


$$
\hat{f}(x_0) = \frac{\Sigma_{i=1}^N K_\lambda(x_0, \ x_i)y_i}{\Sigma_{i=1}^NK_\lambda(x_0, \ x_i)}
$$


where kernel function


$$
K_\lambda(x_0,\ x_i)= D(\frac{|x-x_0|}{\lambda})
$$


We can use different kernel 





* Epanechnikov quadratic kernel


$$
D(t)= \frac{3}{4}(1-t^2) \ if \ |t|<1, \ 0 \ otherwise
$$


* tri-cube function


$$
D(t)=(1-|t|^3)^3 \ if \ |t| \leq 1 , \ 0 \ otherwise
$$




* Gaussian density function


$$
D(t)=\phi(t)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}t^2)
$$






Then, the fitted function is now continuous, and quite smooth

<br/>



**General Notation of kernel function**

Let $h_\lambda(x_0)$ be a width function(indexed by $\lambda$) that determines the width of the neighborhood at $x_0$. Then


$$
K_\lambda(x_0, x) = D(\frac{|x-x_0|}{h_\lambda(x_0)})
$$


In Nadaraya-watson kernel-weighted average, 


$$
h_\lambda(x_0)=\lambda
$$


In KNN


$$
h_k(x_0)=|x_0-x_k|
$$


where $x_k$ : $k$th closest $x_i$ to $x_0$



<br/>

**Details to practice**



* The smoothing parameter $\lambda$, which determines the width of the local neighborhood, has to be determined.

  High $\lambda$ : Low variance, but high bias

  Low $\lambda$ : High variance, but low bias



* Metric window widths (constant $h_\lambda(x)$) tend to keep the bias of the estimate constant, but the variance is inversely proportional to the local density. Nearest-neighbor window widths exhibit the opposite behavior; the variance stays constant and the absolute bias varies inversely with local density.



* $x$값이 같은 여러 observation이 존재할 때 문제점 발생. - $x$값이 같은 observation의 $y$값 평균 내서 새로운 data로 대체 후 weight을 주어 계산  - weight은 kernal weights를 주는 경우가 많음



* Boundary issues arise : Boundary의 경우 한 side의 data로만 추정하기 때문에, 구간 내 데이터가 상대적으로 적음







<br/>

### 2) Local Linear Regression

<br/>



 Nadaraya-watson kernel-weighted average에서의 문제점 : boundary issue!



Solution : 약간의 restriction을 통해 boundary issue를 해결하자!



How : Fitting straight line rather than constants locally, we can remove this bias exactly to first order.



<br/>



**Locally weighted regression**



At each target point $x_0$, solve


$$
\min_{\alpha(x_0), \ \beta(x_0)} \Sigma_{i=1}^N K_\lambda(x_0, x_i)[y_i-\alpha(x_0)-\beta(x_0)x_i]^2 \cdots (1)
$$


Then,


$$
\hat{f}(x_0)=\hat{\alpha}(x_0)+\hat{\beta}(x_0)x_0
$$




* Generalization using matrix



Let 


$$
X = \begin{bmatrix}1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}, \ 
Y=\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}, \
\boldsymbol{\beta}(x_0) = \begin{bmatrix} \alpha(x_0) \\ \beta(x_0) \end{bmatrix}, \ 
\boldsymbol{b}(x_0)=\begin{bmatrix}1 & x_0\end{bmatrix}
$$


and 


$$
W(x_0) \ : n\times n \ diagonal \ matrix \ with \ element \\
W_{ii}(x_0)=K_\lambda(x_i, x_0)
$$


Then, solving (1) is equivalent to solving


$$
\min_{\boldsymbol{\beta}(x_0)}W(x_0)(Y-X\boldsymbol{\beta}(x_0))
$$


which is weighted least square



The solution is


$$
\hat{\boldsymbol{\beta}}(x_0) = (X'W(x_0)X)^{-1}X'W(x_0)Y
$$


and 


$$
\begin{aligned}

\hat{f}(x_0) &= \boldsymbol{b}(x_0)'\hat{\boldsymbol{\beta}}(x_0) \\
&=\Sigma_{i=1}^Nl_i(x_0)y_i
\end{aligned}
$$




The estimate is linear in the $y_i$. $l_i(x_0)$ is referred as the equivalent kernel.



Local linear regression automatically modifies the kernel to correct the bias exactly to first order, a phenomenon dubbed as automatic kernel carpentry.



* 설명


$$
\begin{aligned}

E(\hat{f}(x_0)) &=\Sigma_{i=1}^Nl_i(x_0)f(x_i) \\
&= f(x_0)\Sigma_{i=1}^Nl_i(x_0) + f'(x_0)\Sigma_{i=1}^N(x_i-x_0)l_i(x_0) + \frac{f''(x_0)}{2}\Sigma_{i=1}^N(x_i-x_0)^2l_i(x_0)+R

\end{aligned}
$$


(Taylor expansion is used, $R$ involves third- and higher-order derivatives of $f$, and typically small under suitable smoothness assumptions)



Then,


$$
\Sigma_{i=1}^Nl_i(x_0)=1, \ \Sigma_{i=1}^N(x_i-x_0)l_i(x_0)=0
$$


Therefore


$$
E(\hat{f}(x_0)) = f(x_0)
$$


and bias is 


$$
E(\hat{f}(x_0))-f(x_0)=\frac{f''(x_0)}{2}\Sigma_{i=1}^N(x_i-x_0)^2l_i(x_0)+R
$$


which depends only on quadratic and higher-order terms in the expansion of $f$







<br/>



### 3) Local polynomial regression



<br/>



Fit local polynomial fits of any degree $d$,


$$
\min_{\alpha(x_0), \ \beta_j(x_0), \ j=1, ..., d}\Sigma_{i=1}^NK_\lambda(x_0, x_i)[y_i-\alpha(x_0)-\Sigma_{j=1}^d\beta_j(x_0)x_i^j]^2
$$


and solving this problem, we can get


$$
\hat{f}(x_0)=\hat{\alpha}(x_0)+\Sigma_{j=1}^d\hat{\beta_j}(x_0)x_0^j
$$




Using this method, the bias will only have components of degree $d+1$ and higher, however, the variance is increased.



When we assume


$$
y_i=f(x_i)+\epsilon_i
$$


with $\epsilon_i \sim N(0, \sigma^2)$, iid. Then


$$
Var(\hat{f}(x_0))=\sigma^2||l(x_0)||^2
$$


As degree increases, $||l(x_0)||$ increases. 



**There is a bias-variance tradeoff in selecting the polynomial degree**



<br/>

**Details to practice**



* Local linear fits can help bias dramatically at the boundaries at a modest cost in variance. Local quadratic fits do little at the boundaries for bias, but increase the variance a lot.
* Local quadratic fits tend to be most helpful in reducing bias dueto curvature in the interior of the domain
* Aympototic analysis suggest that local polynomials of odd degree dominate those of even degree. This is largely due to the fact that aysymptotically the MSE is dominated by boundary effects.







<br/>



### 4) Selecting the Width of the Kernel 



<br/>



In each of the kernels $K_{\lambda}$, $\lambda$ is a parameter that controls its width



* For the Epanechnikov or tri-cube kernel with metric width, $\lambda$ is the radius of the support region
* For the GAussian kernel, $\lambda$ is the standard deviation
* $\lambda$ is the number $k$ of nearest neighbors in $k$-nearest neighborhoods, often expressed as a fraction or span $k\ N$

 of the total training sample.



<br/>



**bandwidth** $\lambda$ **and bias-variance tradeoff**



* If the window is narrow, $\hat{f}(x_0)$ is an average of a small number of $y_i$ close to $x_0$

  window에 있는 data 수가 적기 때문에, lower bias, higher variance

* If the window is wide,

  window에 있는 data 수가 많기 때문에, lower variance, higher bias



* In local regression, if width goes to zero, the estimates approach a piecewise-linear function that interpolates th training data
* If width goes to infinity, the estimates approach the global linear least-squares fit to the data.



<br/>

**Methods to find the best** $\lambda$ : Cross validation



LOOCV

GCV

K-fold CV



<br/>



### 5) Local Regression in $\mathbb{R}^p$ 



<br/>



Let $b(X)$ be a vector of polynomial terms in $X$ of maximum degree $d$.

$p$ : number of predictors



Then, in $\mathbb{R}^p$, criteria of local regression is


$$
\min_{\boldsymbol{\beta}(\boldsymbol{x_0})}\Sigma_{i=1}^NK_\lambda(\boldsymbol{x_0}, \boldsymbol{x_i})(y_i-b(\boldsymbol{x_i})'\boldsymbol{\beta}(\boldsymbol{x_0}))
$$




And we can fit


$$
\hat{f}(\boldsymbol{x_0}) = b(\boldsymbol{x_0})'\boldsymbol{\beta}(\boldsymbol{x_0})
$$




Typically, the kernel will be a radial function, such as the radial Epanechnikov or tri-cube kernel


$$
K_\lambda(\boldsymbol{x_0}, \boldsymbol{x})=D(\frac{||\boldsymbol{x}-\boldsymbol{x_0}||}{\lambda})
$$


where $||\cdot||$ is Euclidean norm. We need to standardize each predictor, because of Euclidean norm.



<br/>

**Problems when dimension is high**



* Much bigger boundary problems

  : $p$가 커질 수록, boundary 수가 많아짐.

* Curse of dimensionality

  : 근방에 존재하는 data 수가 dimension이 많아질수록 적어짐 - total size가 증가하지 않는이상, lower bias와 lower variance를 유지하기 어려워짐.

* Visualization

  : 3차원 이상의 data에 대해서 visualization이 힘듦. 

  : 다른 변수를 condition한 후, 두 변수간 visualization이 가능함. but conditioning으로 인한 localization이 됨.







<br/>



### 6) Local likelihood



<br/>



Any parametric model can be made local if the fitting method accommodates observation weights



*example*



* Associated with each observation $y_i$ is a parameter $\theta_i=\theta(\boldsymbol{x_i})=\boldsymbol{x_i}'\boldsymbol{\beta}$ linear in the covariates $x_i$, and inference for $\beta$ is based on the log-likelihood $l(\boldsymbol{\beta})=\Sigma_{i=1}^Nl(y_i,\boldsymbol{x_i}'\boldsymbol{\beta} )$ . We can model $\theta(X)$ more flexibly by using the likelihood local to $x_0$ for inference of $\theta(\boldsymbol{x_0})=\boldsymbol{x_0}'\boldsymbol{\beta}(\boldsymbol{x_0})$


$$
l(\boldsymbol{\beta}(\boldsymbol{x_0}))=\Sigma_{i=1}^NK_{\lambda}(\boldsymbol{x_0}, \boldsymbol{x_i})l(y_i, \boldsymbol{x_i}'\boldsymbol{\beta}(\boldsymbol{x_0}))
$$




Local likelihood allows a relaxation from a globally linear model to one that is locally linear.



* As above, except different variables are associated with $\theta$ from those used for defining the local likelihood:


$$
l(\theta(z_0))=\Sigma_{i=1}^NK_{\lambda}(z_0, z_i)l(y_i, \eta(x_i, \theta(z_0)))
$$




* Local version of the multiclass linear logistic regression model



Features : $x_i$

Associated categorical response : $g_i \in \{1, 2, ..., J\}



Then, the multinomial logit model is for category $j$ is 


$$
P(G=j | X=x) = \frac{\exp(\beta_{j0}+\beta_j'x)}{1+\Sigma_{k=1}^{J-1}\exp(\beta_{k0}+\beta_k'x)}
$$


The local log-likelihood for this $J$ class model can be written


$$
\Sigma_{i=1}^NK_{\lambda}(x_0, x_i)\{\beta_{g_i0}(x_0)+\beta_{g_i(x_0)}'(x_i-x_0) - \log\{1+\Sigma_{k=1}^{J-1}\exp(\beta_{k0}(x_0)+\beta_k(x_0)'(x_i-x_0))\}
$$


Notice that



* We have used $g_i$ as a subscript in the first line to pick out the appropriate numerator($g_i=j$인 obs 표현)
* $\beta_{J0}=0$ and $\beta_J=0$ by the definition of the model(restriction)
* Centered the local regressions at $x_0$, so that the fitted posterior probabilities at $x_0$ are simply


$$
\hat{P}(G=j | X=x_0) = \frac{\exp{(\hat{\beta}_{j0}(x_0)})}{1+\Sigma_{k=1}^{J-1}\exp{(\hat{\beta_{k0}}(x_0)})}
$$




<br/>



### 7) Appendix



<br/>



#### (1) Calculating bias



<br/>



If we set the local linear regression, then the bias of $\hat{f}(x_0)$ consists of second and over order terms. To prove this, we need to show


$$
\Sigma_{i=1}^Nl_i(x_0)=1, \ \Sigma_{i=1}^N(x_i-x_0)l_i(x_0)=0
$$

In fact, in $d$th degree


$$
\Sigma_{i=1}^N(x_i-x_0)^jl_i(x_0)=0
$$


for $j=1, ..., d$



* **Proof**



Let


$$
X=\begin{bmatrix}\boldsymbol{1} & \boldsymbol{v_1} & ... & \boldsymbol{v_d} \end{bmatrix}, \ 
Y=\begin{bmatrix}y_1 \\ y_2 \\ \vdots \\ y_n \ \end{bmatrix}, \
\boldsymbol{\beta}(x_0)=\begin{bmatrix}\beta_0(x_0) \\ \beta_1(x_0) \\ \vdots \\ \beta_d(x_0) \ \end{bmatrix}, \
\boldsymbol{b}(x_0)=\begin{bmatrix}1 \\ x_0 \\ x_0^2 \\ \vdots \\ x_0^d \end{bmatrix}
$$


where


$$
\boldsymbol{v_j}=\begin{bmatrix}x_1^j \\ x_2^j \\ \vdots \\ x_n^j \end{bmatrix}
$$


and let


$$
W(x_0) \ : n\times n \ diagonal \ matrix \ with \ element \\
W_{ii}(x_0)=K_\lambda(x_i, x_0)
$$


Then,


$$
\hat{\boldsymbol{\beta}}(x_0) = (X'W(x_0)X)^{-1}X'W(x_0)Y \\
\hat{f}(x_0) =  \boldsymbol{b}(x_0)'(X'W(x_0)X)^{-1}X'W(x_0)Y=\Sigma_{i=1}^nl_i(x_0)y_i
$$


Instead of $Y$, when we put $X$


$$
\boldsymbol{b}(x_0)'(X'W(x_0)X)^{-1}X'W(x_0)X=\boldsymbol{b}(x_0)' = \begin{bmatrix}1 & x_0 & x_0^2 & ... & x_0^d \end{bmatrix}
$$


It means


$$
\Sigma_{i=1}^nl_i(x_0)x_i^j=x_0^j
$$


When $j=0$ and $j=1$, we get


$$
\Sigma_{i=1}^nl_i(x_0)=1, \ \Sigma_{i=1}^nl_i(x_0)x_i=x_0
$$


Using this facts, we get


$$
\Sigma_{i=1}^n(x_i-x_0)l_i(x_0)= \Sigma_{i=1}^nx_il_i(x_0) - x_0\Sigma_{i=1}^nl_i(x_0)=x_0-x_0=0
$$


Also, in dth degree local polynomial regression


$$
\begin{aligned}
\Sigma_{i=1}^n(x_i-x_0)^jl_i(x_0) &= \Sigma_{i=1}^n\Sigma_{r=1}^{j}  {j \choose r}x_i^r(-x_0)^{j-r}l_i(x_0)\\
&= \Sigma_{r=1}^j{j \choose r}(-x_0)^{j-r}\Sigma_{i=1}^{n}x_i^rl_i(x_0) \\
&= \Sigma_{r=1}^j{j \choose r}(-x_0)^{j-r}x_0^r \\
&= (x_0-x_0)^j=0
\end{aligned}
$$


Therefore, in dth local polynomial regression the bias consists of d+1 and over degree errors.



<br/>



#### (2) Local likelihood - multinomial case

<br/>

By using Nadaraya-Watson kernel smoother, we get


$$
\hat{P}(G=j | X=x) = \frac{\Sigma_{i=1}^NK_\lambda(x_0, x_i)y_i}{\Sigma_{i=1}^NK_\lambda(x_0, x_i)} \propto \Sigma_{i \in G_j}K_\lambda(x_0, x_i)
$$


where $G_j$ is the set of observations which have outcome as $j$th category



In multinomial logit model, we get


$$
P(G=j |X=x) = \frac{\exp(\beta_{j0}+x'\beta_j)}{1+\Sigma_{k=1}^{J-1}\exp(\beta_{j0}+x'\beta_k)}
$$


and the local log likelihood is


$$
\begin{aligned}

l(\boldsymbol{\beta}, x_0) &= \Sigma_{i=1}^NK_\lambda(x_0, x_i)[\beta_{g_i,0}(x_0) + \beta_{g_i}(x_0)'(x_i-x_0) - \log\{1+\Sigma_{k=1}^{K-1}\exp(\beta_{k0}+\beta_{k}(x_0)'(x_i-x_0))    \} ] \\

&=  \Sigma_{i=1}^NK_\lambda(x_0, x_i)[\Sigma_{k=1}^{J-1}I(y_i=k)(\beta_{k_0}(x_0)+\beta_{k}(x_0)'(x_i-x_0))  - \log\{1+\Sigma_{k=1}^{K-1}\exp(\beta_{k0}+\beta_{k}(x_0)'(x_i-x_0))    \}]

\end{aligned}
$$


Because we centered at $x=x_0$, 


$$
\hat{P}(G=j |X=x)=\frac{\exp(\hat{\beta}_{j0})}{1+\Sigma_{k=1}^{J-1}\exp(\hat{\beta}_{j0})}
$$


So, we need $\frac{\partial l(\boldsymbol{\beta}, x_0)}{\beta_{j0}}$


$$
\frac{\partial l(\boldsymbol{\beta}, x_0)}{\beta_{j0}} =\Sigma_{i=1}^NK_\lambda(x_0, x_i)(I(y_i=j) - \frac{\exp(\beta_{j0}+\beta_j(x_0)'(x_i-x_0))}{1+\Sigma_{k=1}^{K-1}\exp(\beta_{k0}+\beta_{k}(x_0)'(x_i-x_0))}) = 0
$$


We can see that


$$
\Sigma_{i=1}^NK_\lambda(x_0, x_i)I(y_i=j) = \Sigma_{i\in G_j}K_\lambda(x_0, x_i) =e^{\beta_{j0}}\Sigma_{i=1}^NK_\lambda(x_0, x_i)\frac{\exp(\beta_j(x_0)'(x_i-x_0))}{1+\Sigma_{k=1}^{K-1}\exp(\beta_{k0}+\beta_{k}(x_0)'(x_i-x_0))}
$$


Therefore, we get


$$
e^{\beta_{j0}} \propto \Sigma_{i\in G_j}K_\lambda(x_0, x_i)
$$
