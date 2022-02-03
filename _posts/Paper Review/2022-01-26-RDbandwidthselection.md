

## Paper Review : Optimal Bandwidth Choice for the Regression discontinuity Estimator



#### 1. Basic model

<br/> 

**Potential outcome framework**



Notation



Sample size : $N$

$Y_i(1)$ : potential outcome for unit $i$ given treatment

$Y_i(0)$ : potential outcome for unit $i$ without treatment

$W_i$ : Whether the treatment received or not. $W_i=1$ : treatment received, $W_i=0$ : not received



Then, the observed outcome $Y_i$ is


$$
Y_i=Y_i(W_i)=Y_i(0) \ \ if \ \ W_i=0 \\
Y_i=Y_i(W_i)=Y_i(1) \ \ if \ \ W_i=1 \\\\

Y_i=W_iY_i(1)+(1-W_i)Y_i(0)
$$




<br/>

**Regression discontinuity design**





$X_i$ : Forcing variable with scalar covariate. This variable determines the treatment



$m(x)$ : conditional expectation of $Y_i$ given $X_i=x$


$$
m(x)=E (Y_i | X_i=x)
$$




In SRD design, treatment is determined solely by the value of the forcing variable $X_i$ being on either side of a fixed and known threshold $c$ or


$$
W_i = I\{X_i\geq c\}
$$




Then, we focus on average effect of the treatment for units with covariate values equal to the threshold


$$
\tau_{SRD} = E(Y_{i}(1) - Y_{i}(0) | X_i=c)
$$


If the conditional distribution functions $F_{Y(0)|X}(y|x)$ and $F_{Y(1)|X}(y|x)$ is continuous in $x$  for all $y$ and the conditional first moments $E(Y_i(1) | X_i=x)$ and $E(Y_i(0)|X_i=x)$ exist and are continuous at $x=c$, then


$$
\tau_{SRD} = \mu_{+} - \mu_{-} = \lim_{x\downarrow c}m(x) - \lim_{x\uparrow c}m(x)
$$


The estimand is the difference of two regression functions evaluated at boundary points.



We use local linear regression at each side to estimate $\tau_{SRD}$



Fit local linear regression


$$
(\hat{\alpha}_{-}(x), \hat{\beta}_-(x)) = \arg \min_{\alpha, \beta} \Sigma_{i=1}^NI(X_i<x)(Y_i-\alpha-\beta(X_i-x))^2K(\frac{X_i-x}{h}) \\ \\
(\hat{\alpha}_{+}(x), \hat{\beta}_+(x)) = \arg \min_{\alpha, \beta} \Sigma_{i=1}^NI(X_i\geq x)(Y_i-\alpha-\beta(X_i-x))^2K(\frac{X_i-x}{h})
$$


Then, the estimated regression function $m(\cdot)$ at $x$ is


$$
\hat{m}_h(x) = \hat{\alpha}_-(x) \ \ if \ \ x<c \\
\hat{m}_h(x) = \hat{\alpha}_+(x) \ \ if \ \ x \geq c
$$


Then, the estimated $\tau_{SRD}$ is


$$
\hat{\tau}_{SRD} = \hat{\mu}_+ - \hat{\mu}_-, \ where \\

\hat{\mu}_- = \lim_{x\uparrow c}\hat{m}_h(x) = \hat{\alpha}_-(c), \ \hat{\mu}_+ = \lim_{x\downarrow c}\hat{m}_h(x) = \hat{\alpha}_+(c),
$$

<br/>



### 2. Error Criterion and Infeasible optimal bandwidth choice



<br/>



#### 1) Error criteria

<br/>



optimal choice of the bandwidth $h$에서 사용되었던 기존의 방법론 : cross validation or ad hoc methods.



Cross validation를 적용하기 위해 사용한 식 : mean integrated squared error criterion(MISE)




$$
MISE(h) = E(\int_x (\hat{m}_h(x) - m(x))^2f(x)dx)
$$


where $f(x)$ is forcing variable.



Problem



$MISE(h)$를 이용하여 구한 optimal bandwidth $h$는 최적의 $\hat{\tau}_{SRD}$를 만들어주는 $h$가 아닌 최적의 $\hat{m}_h(x)$를 만들어주는 $h$가 됨. 즉 목적함수가 다름



Property of $\tau_{SRD}$



(1) $\tau_{SRD}$는 $m(x)$ 중 $x=c$에서의 좌극한, 우극한 값만 필요함. 실제로 $m(x)$를 추정할 때 $x=c$를 기준으로 각각의 side의 data를 진행하여 local linear regression을 진행하기 때문에, 시행하는 local linear regression 횟수는 2번, $\tau_{SRD}$ 추정할 때 사용하는 값 역시 2개($\hat{\mu}_-, \hat{\mu}_+$)

(2) 추정하는 두 개의 값 ($\hat{\mu}_-, \hat{\mu}_+$)이 boundary point!



따라서 MISE가 아닌 다른 error criteria를 이용하여 best $h$를 추정해야 함. 즉, $\tau_{SRD}$에 대한 mean squared error를 정의하고


$$
MSE(h) = E((\hat{\tau}_{SRD}-\tau_{SRD})^2) = E(((\hat{\mu}_+ - \mu_+) - (\hat{\mu}_- - \mu_-))^2)
$$


RD design에서의 optimal bandwidth $h$는 위 $MSE(h)$를 최소화시키는 $h$가 됨


$$
h^* = \arg\min_hMSE(h)
$$


Problem



sample size가 커져도, $h^*$가 0으로 converge하지 않는 경우가 발생 

(side별로 estimation을 진행하기 때문에, bias가 상쇄되는 경우가 발생할 수 있음)

It does not seem appropriate to base estimation on global criteria when identification is local



- Focusing on the bandwidth that minimizes a first-order approximation to $MSE(h)$ : Asymptotic mean squared error $AMSE(h)$





Second concern : Single bandwidth



local linear regression을 두번 진행하기 때문에, 각각의 local linear regression에 최적의 bandwidth가 존재할수도 있음, 따라서


$$
MSE(h_-, h_+) = E(((\hat{\mu}_+(h_+) - \mu_+) - (\hat{\mu}_-(h_-) - \mu_-))^2)
$$


를 최소화시키는 $h_-, h_+$를 찾을수도 있음.



Problem



Suppose the bias for both estimators are strictly increasing. Then, we can set $h_+(h_-)$ such that the bias of the RD estimate cancel out.


$$
(E(\hat{\mu}_-(h_-))-\mu_-)-(E(\hat{\mu}_+(h_+(h_-))-\mu_+))=0
$$


 $h_-$ 를 크게 setting한 후,  위의 bias가 0이 되도록 적절한 $h_+$를 찾아주기만 하면 됨. 즉 bandwidth가 무한히 커지더라도, bias가 0이 될 수 있으므로 문제가 발생할 수 있음. 따라서 실제 적용에 문제가 발생할 수 있음!





<br/>



#### 2) An asymptotic expansion of the expected error



<br/>



**Notation**



 $m^{(k)}_+(c)$  : right limits of the $k$th derivative of $m(x)$ at the threshold $c$

 $m^{(k)}_-(c)$ : left limits of the $k$th derivative of $m(x)$ at the threshold $c$

$\sigma^2_+(c)$ : The right limit of the conditional variance $\sigma^2(x)=Var(Y_i | X_i=x)$ at the threshold $c$

$\sigma^2_-(c)$ : The left limit of the conditional variance $\sigma^2(x)=Var(Y_i | X_i=x)$ at the threshold $c$



<br/>



**Assumption**



(1) $(Y_i, X_i), for \ i=1, ..., N$ are iid

(2) The marginal distribution of the forcing variabel $X_i$, denoted $f(\cdot)$, is continuous and bounded away from zero at the threshold $c$

(3) The conditional mean $m(x)=E(Y_i | X_i=x)$ has at least three continuous derivatives in an open neighbourhood of $X=c$. The right and left limits of the $k$th derivative of $m(x)$ at the threshold $c$ are denoted by $m^{(k)}_+(c)$ and $m^{(k)}_-(c)$ .

(4) The kernel $K(\cdot)$ is non-negative, bounded, differs from zero on a compact interval $[0, a]$, and is continuous on $(0, a)$

(5) The conditional variance function $\sigma^2(x) = Var(Y_i | X_i=x)$ is bounded in an open neighbourhood of $X=c$ and right and left continuous at $c$.

(6) The second derivatives from the right and the left differ at the threshold: $m^{(2)}_+(c) \neq m^{(2)}_-(c)$





<br/>



**Definition : **$AMSE(h)$


$$
AMSE(h)=C_1h^4(m^{(2)}_+(c) -m^{(2)}_-(c))^2 + \frac{C_2}{Nh}(\frac{\sigma^2_+(c)}{f(c)} + \frac{\sigma^2_-(c)}{f(c)})
$$


$C_1, C_2$ are functions of the kernel:


$$
C_1=\frac{1}{4}(\frac{\nu_2^2-\nu_1\nu_3}{\nu_2\nu_0-\nu_1^2})^2, \ C_2=\frac{\nu_2^2\pi_0 - 2\nu_1\nu_2\pi_1+\nu_1^2\pi_2}{(\nu_2\nu_0-\nu_1^2)^2}
$$


where 


$$
\nu_j = \int_0^\infty u^jK(u)du , \ \pi_j=\int_0^\infty u^jK^2(u)du
$$




In AMSE, the first term


$$
C_1h^4(m^{(2)}_+(c) -m^{(2)}_-(c))^2
$$


corresponds to the square of the bias and the second term


$$
\frac{C_2}{Nh}(\frac{\sigma^2_+(c)}{f(c)} + \frac{\sigma^2_-(c)}{f(c)})
$$




corresponds to the variance.



In bias term clarifies the role that assumption (6) will play. 



The leading term in the expansion of the bias : order $h^4$ if assumption (6) holds



If the assumption (6) does not hold, the bias converges to zero faster, allowing for estimation for $\tau_{SRD}$ at a faster rate of convergence.



(실제로는 second derivative가 같은지 확인이 어렵기 때문에, assumption (6)을 만족한 상태에서 진행. (6)을 만족하지 않더라도, proposed estimator $\tau_{SRD}$ : constistent)



(second derivative가 같은 경우와 다른 경우 optimal bandwidth를 찾는 방법에서 차이가 있을 수 있음. 이 논문에서는 다른 경우에 대해서 다룸)



<br/>



**Lemma 1 (Mean Squared Error Approximation and Optimal Bandwidth**



(1) Suppose assumptions (1) - (5) hold. Then


$$
MSE(h)=AMSE(h)+o_p(h^4+\frac{1}{Nh})
$$




(2) Suppose that also assumption (6) holds. Then,


$$
h_{opt}=\arg\min_hAMSE(h)=C_k(\frac{\sigma^2_+(c)+\sigma^2_-(c)}{f(c)(m^{(2)}_+(c)-m^{(2)}_-(c))^2})^{1/5}N^{-1/5}
$$


where $C_K=(\frac{C_2}{4C_1})^{1/5}$ , indexed by the kernel $K(\cdot )$





For the edge kernel, with $K(u)=I\{|u|\leq1\}(1-|u|)$, the constant $C_{K, edge} \approx 3.4375$ 

For the uniform kernel with $K(u)=I\{|u|\leq1/2\}$, the constant  $C_{K, uniform} \approx 5.40$



<br/>



### 3. Feasible optimal bandwidth choice



<br/>

#### 1) A simple plug-in bandwidth

<br/>


$$
{h}_{opt}=\arg\min_hAMSE(h)=C_k(\frac{\sigma^2_+(c)+\sigma^2_-(c)}{f(c)(m^{(2)}_+(c)-m^{(2)}_-(c))^2})^{1/5}N^{-1/5}
$$


에서 필요한 값 


$$
\sigma^2_+(c), \sigma^2_-(c), f(c), m^{(2)}_+(c), m^{(2)}_-(c), K(\cdot)
$$


을 해당 unknown quantities의 consistent estimator로 교체


$$
\tilde{h}_{opt}=C_k(\frac{\hat{\sigma}^2_+(c)+\hat{\sigma}^2_-(c)}{\hat{f}(c)(\hat{m}^{(2)}_+(c)-\hat{m}^{(2)}_-(c))^2})^{1/5}N^{-1/5}
$$




problem 



First-order bias가 매우 작을 때 문제가 발생할 수 있음

위 경우 $m^{(2)}_+(x) = m^{(2)}_-(x)$가 발생할 수 있고, 이는 $h_{opt}$ 식에서 분모 값이 매우 큰 값이 나올 수 있음. 이 경우 bandwidth가 부정확하고 variance가 커질 수 있음. 

추가적으로, estimator for $\tau_{SRD}$가 poor property를 가지게 됨. -because the true finite sample bias depends on global properties of the regression function that are not captured by the asymptotic approximation used to calculate the bandwidth.



<br/>



##### (1) Regularization



$h_{opt}$의 분모가 0이 되지 않도록 수치 조절



The bias in the plug-in estimator for the reciprocal of the squared difference in second derivatives is


$$
E(\frac{1}{(\hat{m}^{(2)}_+(c)-\hat{m}^{(2)}_-(c))^2} - \frac{1}{({m}^{(2)}_+(c)-{m}^{(2)}_-(c))^2} ) \\
=(\frac{3(Var(\hat{m}^{(2)}_+(c))+Var(\hat{m}^{(2)}_-(c)))}{(m^{(2)}_+(c)-m^{(2)}_-(c))^4}) + o(N^{-2\alpha})
$$


Then, for $r=3(Var(\hat{m}^{(2)}_+(c))+Var(\hat{m}^{(2)}_-(c)))$, the bias in the modified estimator for the reciprocal of the squared difference in second derivatives in of lower order


$$
E(\frac{1}{(\hat{m}^{(2)}_+(c)-\hat{m}^{(2)}_-(c))^2+r} - \frac{1}{({m}^{(2)}_+(c)-{m}^{(2)}_-(c))^2} ) = o(N^{-2\alpha})
$$


This in turn motivates the modified bandwidth estimator


$$
\hat{h}_{opt} = C_K(\frac{\hat{\sigma}^2_-(c) + \hat{\sigma}^2_+(c)}{\hat{f}(c)((\hat{m}^{(2)}_+(c) - \hat{m}^{(2)}_-(c))^2 +r_+ + r_-)})^\frac{1}{5}N^{-\frac{1}{5}}
$$


where


$$
r_+ = 3 \hat{Var}(\hat{m}_-^{(2)(c)}), r_- = 3\hat{Var}(\hat{m}_+^{(2)}(c))
$$


Then, this bandwidth will not become infinite even in the cases when the difference in curvatures at  the threshold is zero.



<br/>

##### (2) Implementing the regularization



We estimate the second derivative $m_+^{(2)}(c)$ by fitting a quadratic function to the observations with $X_i \in [c, c+h]$.

The initial bandwidth $h$ here will be different from the bandwidth $\hat{h}_{opt}$ used in the estimation of $\tau_{SRD}$



Notation



$N_{h, +}$ : the number of units with covariate values in this interval

$\bar{X} = \frac{1}{N_{h, +}}\Sigma_{c\leq X_i\leq c+h} X_i$

$\hat{\mu}_{j, h, +} = \frac{1}{N_{h, +}}\Sigma_{c\leq X_i\leq c+h}(X_i-\bar{X})^j$ : $j$th centered moment of the $X_i$ in the interval $[c, c+h]$.



Then, we can get $r_+$


$$
r_+ = \frac{12}{N_{h, +}}(\frac{\sigma^2_{+}(c)}{\hat{\mu}_{4, h, +} - (\hat{\mu}_{2, h, +})^2 - (\hat{\mu}_{3, h, +})^2/\hat{\mu}_{2, h, +}})
$$


However, fourth moments are difficult to estimate precisely, we approximate this expression exploiting the fact that for small $h$, the distribution of the forcing variable can be approximated by a uniform distribution on $[c, c+h]$, so that 


$$
\mu_{2, h, +} \approx h^2/12, \ \mu_{3, h, +} \approx 0 , \ \mu_{4, h, +} \approx \frac{h^4}{60}
$$


Using this facts,


$$
\hat{r}_+ = \frac{2160\hat{\sigma}^2_+(c)}{N_{h, +}h^4}, \ \hat{r}_- = \frac{2160\hat{\sigma}^2_-(c)}{N_{h, -}h^4} 
$$


Then using $\hat{r} = \hat{r}_- + \hat{r}_+$, we can get


$$
\hat{h}_{opt} = C_K(\frac{\hat{\sigma}^2_-(c) + \hat{\sigma}^2_+(c)}{\hat{f}(c)((\hat{m}^{(2)}_+(c) - \hat{m}^{(2)}_-(c))^2 +r_+ + r_-)})^\frac{1}{5}N^{-\frac{1}{5}}
$$


* Check

We need specific estimators $\hat{\sigma}^2_+(c), \hat{\sigma}^2_-(c), \hat{f}(c), \hat{m}^{(2)}_+(c), \hat{m}^{(2)}_-(c)$



Any combination of consistent estimators for $\sigma^2_+(c), \sigma^2_-(c), f(c), m^{(2)}_+(c), m^{(2)}_-(c)$ substituted into expression, with or without the regularity terms, will have the same optimality properties



The proposes estimator is relatively simple, but the more important point is that it is a specific estimator: It gives a convenient starting point and benchmark for doing a sensitivity analyses regarding bandwidth choice.



The bandwidth selection algorithm to be relatively robust to these choices.



<br/>

#### 2) An algorithm for bandwidth selection

<br/>



##### (1) Step 1. Estimation of density $f(c)$ and conditional variances $\sigma^2_-(c)$ and $\sigma^2_+(c)$



First, calculate the sample variance of the forcing variable, $S^2_X = \Sigma(X_i-\bar{X})^2/(N-1)$

Use the Silverman rule to get a pilot bandwidth for calculating the density and variance at $c$



For normal kernel and a normal reference density : $h=1.06 S_X N^{-1/5}$

Modification : Uniform kernel on $[-1, 1]$ and normal reference density


$$
h_1= 1.84S_XN^{-1/5}
$$


Then, calculate 


$$
N_{h_{1, -}} = \Sigma_{i=1}^N I\{c-h_1 \leq X_i < c\}, \ N_{h_{1, +}} = \Sigma_{i=1}^N I\{c \leq X_i \leq c+h_1\} \\

\bar{Y}_{h_1, -} = \frac{1}{N_{h_{1, -}}}\Sigma_{c-h_1 \leq X_i < c} Y_i

\bar{Y}_{h_1, +} = \frac{1}{N_{h_1, +}}\Sigma_{c\leq X_i \leq c+h_1}Y_i

$$


Now estimate the density of $X_i$ at $c$ as


$$
\hat{f}(c) = \frac{N_{h_1, -}+N_{h_1, -}}{2Nh_1}
$$


and estimate the limit of the conditional variances of $Y_i$ given $X_i=x$ at $x=c$


$$
\hat{\sigma}_-^2(c) = \frac{1}{N_{h_1, -}-1} \Sigma_{c-h_1 \leq X_i < c}(Y_i-\bar{Y}_{h_1, -})^2 \\
\hat{\sigma}_+^2(c) = \frac{1}{N_{h_1, +}-1} \Sigma_{c \leq X_i \leq c+h_1}(Y_i-\bar{Y}_{h_1, +})^2
$$


these estimators are consistent for the density and the conditional variance, respectively.



##### (2) Step 2. Estimation of second derivatives $\hat{m}_{+}^{(2)}(c)$ and $\hat{m}_{-}^{(2)}(c)$



First, we need pilot bandwidths $h_{2, -}, h_{2, +}$ Fit a third-order polynomial to the data, including an indicator for $X_i\geq 0$ . 


$$
Y_i = \gamma_0 + \gamma_1 I(X_i\geq c) + \gamma_2(X_i-c) + \gamma_3(X_i-c)^2 + \gamma_4(X_i-c)^3 + \epsilon_i
$$


and estimate  $m^{(3)}(C)$ as $\hat{m}^{(3)}(c) = 6\hat{\gamma}_4$

Note that $\hat{m}^{(3)}(c)$ is in general not a consistent estimate of $m^{(3)}(c)$ but will converge to some constant at a parametric rate. However, we do not need a consistent estimate of the third derivative at $c$ here to obtain consistent estimator for the second derivative.



Calculate $h_{2, +}, h_{2, -}$


$$
h_{2, +} = 3.56 (\frac{\hat{\sigma}_+^2(c)}{\hat{f}(c)(\hat{m}^{(3)}(c))^2})^{1/7}N_+^{-1/7}\\

h_{2, -} = 3.56 (\frac{\hat{\sigma}_-^2(c)}{\hat{f}(c)(\hat{m}^{(3)}(c))^2})^{1/7}N_+^{-1/7}
$$


Where $N_-$ and $N_+$ are the number of observations to the left and right of the threshold, respectively.



$h_{2, -}, h_{2, +}$ are estimates of the optimal bandwidth for calculation of the second derivative at a boundary point using a local quadratic and a uniform kernel.



Given the pilot bandwidth $h_{2, +}$, we estimate the curvature $m_+^{(2)}(c)$ by a local quadratic fit. To be precise, temporarily discard the observations other than the $N_{2, +}$ oservations with $c \leq X_i \leq c+h_{2, +} $. 



Label the new data


$$
\boldsymbol{\hat{Y}}_+ = (Y_1, ..., Y_{N_2, +}), \ \boldsymbol{\hat{X}}_+ = (X_1, ..., X_{N_2, +}) \\
\boldsymbol{T} = \begin{bmatrix} 1 & \boldsymbol{T}_1 & \boldsymbol{T}_2  \end{bmatrix}
$$


where $\boldsymbol{T'}_j = ((X_1-c)^j, ..., (X_{N_{2, +}}-c)^j)$



The estimated regression coefficients are


$$
\hat{\lambda} = (\boldsymbol{T'}\boldsymbol{T})^{-1}\boldsymbol{T'}\boldsymbol{\hat{Y}}
$$


and calculate $\hat{m}_{+}^{2}(c) = 2\hat{\lambda}_3$



Similarly, we can calculate $\hat{m}_{-}^{2}(c)$



##### (3) Step 3. Calculation of regularization term $\hat{r}_-$ and $\hat{r}_+$ and calculation of $\hat{h}_{opt}$

Given the previous steps, the regularization terms are calculated as follows:


$$
\hat{r}_+ = \frac{2160 \hat{\sigma}^2_+(c)}{N_{2, +}h^4_{2, +}}, \ \hat{r}_- = \frac{2160 \hat{\sigma}^2_-(c)}{N_{2, -}h^4_{2, -}}, 
$$


Then finally, we can get the proposed bandwidth:


$$
\hat{h}_{opt} = C_K(\frac{\hat{\sigma}^2_-(c) + \hat{\sigma}^2_+(c)}{\hat{f}(c)((\hat{m}^{(2)}_+(c) - \hat{m}^{(2)}_-(c))^2 +r_+ + r_-)})^\frac{1}{5}N^{-\frac{1}{5}}
$$


Given the bandwidth $\hat{h}_{opt}$, we get


$$
\hat{\tau}_{SRD} = \lim_{x\downarrow c}\hat{m}_{\hat{h}_{opt}}(x) - \lim_{x\uparrow c}\hat{m}_{\hat{h}_{opt}}(x)
$$


where $\hat{m}_h(x)$ is the local linear regression estimator.



 ##### (3) Properties of algorithm



First, the resulting RD estimator $\hat{\tau}_{SRD}$ is consistent at the best rate for non-parametric regression functions at a point.

Second, the estimated constant term in the reference bandwidth converges to the best constant.

Third, we have a Li type optimality result for the mean squared error and consistency at the optimal rate for the RD estimate.



**Theorem : Properties of $\hat{h}_{opt}$**



Suppose assumptions (1) - (5) hold. Then:



(1) consistency : If assumption (6) hold. then,


$$
\hat{\tau}_{SRD} - \tau_{SRD} = O_p(N^{-2/5})
$$


(2) consistency : If assumption (6) does not hold, then


$$
\hat{\tau}_{SRD} - \tau_{SRD} = O_p(N^{-3/7})
$$


(3) convergence of bandwidth


$$
\frac{\hat{h}_{opt} - h_{opt}}{h_{opt}} = o_p(1)
$$




(4) Li's optimality


$$
\frac{MSE(\hat{h}_{opt}) - MSE(h_{opt})}{MSE(h_{opt})} = o_p(1)
$$


If assumption (6) does not hold, there can be


$$
m^{(2)}_+(x) = m^{(2)}_-(x)
$$


implying that the bias term of AMSE vanishes, which would improve convergence.





##### (4) DesJardins-McCall bandwidth selection



The objective criterion is different


$$
E((\hat{\mu}_+ - \mu_+)^2 + (\hat{\mu}_- - \mu_-)^2)
$$


The single optimal bandwidth based on the DesJardins and McCall criterion is


$$
h_{DM} = C_K(\frac{\sigma_{+}^2(c) + \sigma^2_{-}(c)}{f(c)(m^{2}_+(c)^2 + m^{(2)}_{-}(c)^2)})^{1/5}N^{-1/5}
$$


This will in large samples lead to a smaller bandwidth than our proposed bandwidth choice if the second derivatives are of the same sign. Also, this model actually use different bandwidths on the left and the right and also use a Epancechnikov kernel.





##### (5) Ludwig-Miller cross-validation



Let $N_-$ and $N_+$ be the number of observations with $X_i<c$ and $X_i \geq c$. for $\delta \in (0, 1)$, let $\theta_-(\delta)$ and $\theta_+(\delta)$ be the $\delta$th quantile of the $X_i$ among the subsample of observations with $X_i<c$ and $X_i\geq c$, respectively, so that


$$
\theta_-(\delta) = \arg\min_a \{a | (\Sigma_{i=1}^nI\{X_i\leq a\} \geq \delta N_- \} \\

\theta_+(\delta) = \arg\min_a \{a | (\Sigma_{i=1}^nI\{c \leq X_i \leq a\} \geq \delta N_+ \}
$$


Not the LM cross-validation criterion we use is of the form


$$
CV_\delta(h) = \Sigma_{i=1}^N I\{\theta_-(1-\delta) \leq X_i \leq \theta_+(\delta)\}(Y_i-\hat{m}_h(X_i))^2
$$


Key feature of $\hat{m}_h(x)$ is that for values of $x<c$, it only uses observations with $X_i<x$ to estimate $m(x)$ and for values of $x \geq c$, it only uses observations with $X_i>x$ to estimate $m(x)$, so that $\hat{m}_h(X_i)$ does not depend on $Y_i$, as is necessary for cross validation.

By using a value for $\delta$ close to zero, we only use observations close to the threshold to evaluate the cross-validation criterion.



Issue

by using LM cross-validation, the criterion focuses on minimizing 


$$
E((\hat{\mu}_+ - \mu_+)^2+ (\hat{\mu}_--\mu_-)^2)
$$


rather than


$$
E(((\hat{\mu}_+ -\hat{\mu}_-) -(\mu_+ -\mu_-) )^2)
$$


Therefore, even letting $\delta \rightarrow 0$ with the sample size in the cross-validation procedure will not result in an optimal bandwidth.



<br/>

### 5. Extension



<br/>



#### 1) Fuzzy regression design



<br/>



In FRD design, the treatment $W_i$ is not a deterministic function of the forcing variable. Instead, the probability $P(W_i=1 | X_i=x)$ changes discontinuously at the threshold $c$. in FRD design, the treatment effect is


$$
\tau_{FRD} = \frac{\lim_{x\downarrow c}E(Y_i | X_i=x) - \lim_{x \uparrow c}E(Y_i | X_i=x)}{\lim_{x\downarrow c}E(W_i | X_i=x) - \lim_{x\uparrow c}E(W_i | X_i=x)}
$$


In this case, we need to estimate two regression functions, each at two boundary points 

: The expected outcome given the forcing variable $E(Y_i | X_i=x)$ to the right and left of the threshold $c$

: The expected value of the treatment variable given the forcing variable $E(W_i | X_i=x)$ to the right and left of $c$



Define


$$
\tau_{Y} = \lim_{x\downarrow c}E(Y_i | X_i=x) - \lim_{x \uparrow c}E(Y_i | X_i=x), \\

\tau_{W} = \lim_{x\downarrow c}E(W_i | X_i=x) - \lim_{x\uparrow c}E(W_i | X_i=x)
$$




with $\hat{\tau}_Y$, $\hat{\tau}_W$ denoting the corresponding estimators, so that 


$$
\tau_{FRD} = \frac{\tau_Y}{\tau_W}, \ \hat{\tau}_{FRD} = \frac{\hat{\tau}_Y}{\hat{\tau}_W}
$$
Then, we can approximate the difference $\hat{\tau}_{FRD} - \tau_{FRD}$ by


$$
\hat{\tau}_{FRD} - \tau_{FRD} = \frac{1}{\tau_W}(\hat{\tau}_{Y} - \tau_{Y}) - \frac{\tau_Y}{\tau_W^2}(\hat{\tau}_W-\tau_W) + o_p((\hat{\tau}_Y-\tau_Y) + (\hat{\tau}_w - \tau_w))
$$


This is the basis for the asymptotic approximation to the MSE around $h=0$


$$
AMSE_{FRD}(h) = C_1h^4(\frac{1}{\tau_W}(m^{(2)}_{Y, +}(c) - m^{(2)}_{Y, -}(c)) - \frac{\tau_Y}{\tau_W^2}(m^{(2)}_{W, +}(c) - m^{(2)}_{W, -}(c)))^2 \\
+ \frac{C_2}{Nhf(c)}(\frac{1}{\tau_W^2}(\sigma^2_{Y, +}(c) + \sigma^2_{Y, -}(c)) + \frac{\tau_Y^2}{\tau_W^4}(\sigma^2_{W, +}(c)+\sigma^2_{W, -}(c)) - \frac{2\tau_Y}{\tau^3_W}(\sigma_{YW, +}(c)+\sigma_{YW, -}(c)))
$$


$C_1, C_2$ are functions of the kernel:


$$
C_1=\frac{1}{4}(\frac{\nu_2^2-\nu_1\nu_3}{\nu_2\nu_0-\nu_1^2})^2, \ C_2=\frac{\nu_2^2\pi_0 - 2\nu_1\nu_2\pi_1+\nu_1^2\pi_2}{(\nu_2\nu_0-\nu_1^2)^2}
$$


where 


$$
\nu_j = \int_0^\infty u^jK(u)du , \ \pi_j=\int_0^\infty u^jK^2(u)du
$$



Difference between SRD and FRD is the addition of probability of treatment variable, therefore we need to consider the variance term of $W_i$ and covariance of $W_i, Y_i$



The bandwidth that minimizes the AMSE in the fuzzy design is


$$
h_{opt, RFRD} = C_K N^{-1/5} \times \\
(\frac{(\sigma^2_{Y, +}(c)+\sigma^2_{Y, -}(c)) + \tau^2_{FRD}(\sigma^2_{W, +}(c)+\sigma^2_{W, -}(c)) -2\tau_{FRD}(\sigma_{YW, +}(c) + \sigma_{YW, -}(c))}{f(c)((m^{(2)}_{Y, +}(c)-m^{(2)}_{Y, -}(c)) - \tau_{FRD}(m^{(2)}_{W, +}(c) - m^{(2)}_{W, -}(c)))^2})^{1/5}
$$


The analogue of the bandwidth proposed for the SRD is



​	
$$
\hat{h}_{opt, RFRD} = C_K N^{-1/5} \times \\
(\frac{(\hat{\sigma}^2_{Y, +}(c)+\hat{\sigma}^2_{Y, -}(c)) + \hat{\tau}^2_{FRD}(\hat{\sigma}^2_{W, +}(c)+\hat{\sigma}^2_{W, -}(c)) -2\hat{\tau}_{FRD}(\hat{\sigma}_{YW, +}(c) + \hat{\sigma}_{YW, -}(c))}{\hat{f}(c)((\hat{m}^{(2)}_{Y, +}(c)-\hat{m}^{(2)}_{Y, -}(c)) - \hat{\tau}_{FRD}(\hat{m}^{(2)}_{W, +}(c) - \hat{m}^{(2)}_{W, -}(c)))^2 + \hat{r}_{Y, +} + \hat{r}_{Y, -} + \hat{\tau}_{FRD}(\hat{r}_{W, +} + \hat{r}_{W, -})})^{1/5}
$$


Implementation



First, using the algorithm described for the SRD case separately for the treatment indicator and the outcome, calculate


$$
\hat{\tau}_{FRD}, \hat{f}(c), \hat{\sigma}^2_{Y, +}, \hat{\sigma}^2_{Y, -}, \hat{\sigma}^2_{W, +}, \hat{\sigma}^2_{W, -}, \hat{m}^{(2)}_{Y, +}(c), \hat{m}^{(2)}_{Y, -}(c), \hat{m}^{(2)}_{W, +}(c), \hat{m}^{(2)}_{W, -}(c), \hat{r}_{Y, +}, \hat{r}_{Y, -}, \hat{r}_{W, +}, \hat{r}_{W, -}
$$


Second, using the initial Silverman bandwidth, use the deviations from the means to estimate the conditional covariances $\hat{\sigma}_{YW, +}(c), \hat{\sigma}_{YW, -}(c)$

Then substitute everything into the expression for the bandwidth.



In practice, this often leads to bandwidth choices similar to those based on the optimal bandwidth for estimation of only the numerator of the RD estimand. One may therefore simply wish to use the basic algorithm ignoring the fact that the regression discontinuity design is fuzzy.



<br/>



#### 2) Additional covariates



<br/>



The presence of additional covariates does not affect the RD analyses very much. If the distribution of the additional covariates does not exhibit any discontinuity around the threshold for the forcing variable, and as a result, those covariates are approximately independent of the treatment indicator for smaples constructed to be close to the threshold.

In that case, the covariates only affect the precision of the estimator, and one can modify the previous analysis using the conditional variance of $Y_i$ given all covariates at the threshold, $\sigma^2_-(c|x)$ and $\sigma^2_+(c|x)$ instead of the variances $\sigma^2_-(c)$ and $\sigma^2_+(c)$ that condition only on the forcing variable.

In practice, this modification does not affect the optimal bandwidth much unless the additional covariates have great explanatory power, and the basic algorithm is likely to perform adequately even in the presence of covariates.











정리



RD design에서 local linear regression을 적용할 때, 선택해야 할 parameter가 bandwidth h

이전에는 기존의 local linear regression에서의 bandwidth selection처럼, MISE를 minimize하는 h를 이용하여 RD design에 적용하였음



하지만 MISE를 criteria로 하여 찾은 optimal bandwidth h는 m(x) 함수(local linear estimator) 자체를 best하게 만들어주는 h

이를 그대로 RD design에 적용하는데는 문제가 있음



RD design에서 추정해야 하는 값과, local linear regression에서 추정해야 하는 값이 다르다! RD design에서는 cutoff에서의 추정값만을 사용하기 때문에, 전체 함수에 대해서 best하게 만들어주는 bandwidth가 아닌, cutoff에서의 추정값, 더 정확히는 tau_SRD를 best하게 추정해주는 h를 찾아야 한다



tau_SRD가 조금 특별한 값 - boundary point 



위 두 문제 때문에 기존의 local linear regression 방법에서 사용되었던 bandwidth selection은 문제가 있다!



어떻게 해결했어?



tau_SRD에 대한 MSE를 정의 하고, 위 MSE 비슷한 AMSE를 minimize시키는 h를 최적의 bandwidth라고 하자!



AMSE 해석



첫번째 텀 : bias^2텀

구성 : m+^2(c)의 bias와 m-^2(c)의 bias로 이루어져 있음

두번째 텀 : variance 텀

구성 : m+^2(c)의 variance와 m-^2(c)의 variance로 이루어져 있음

AMSE가 MSE랑 많이 비슷해서 

AMSE를 minimize하는 h가 최적의 h



----- 실제 estimation



optimal h 식에서 우리가 모르는 값이 6개 



Ck - kernel function select하면 결정

나머지 모르는 값 - consistent하게만 정해주면 결과가 일치한다



문제점 : m(c)의 second derivative가 비슷하면 문제가 발생함(bandwidth가 무한히 커질 수 있지)



해결방법 : regularization : 분모항 bias에서 착안, 결과 분자항을 더해주면 error 작게 나오면서 위의 문제를 해결할 수 있음 (왜 분모에 더하지??? 조금 더 생각) 



r도 사실은 몰라요 : 왜냐면 m(x)를 모르니까

해결 : quadratic regression 이용하여 추정함 - 너무 복잡해서 approximation 사용함



실제로 어떻게 하냐 - 3 step



1. f(c), sigma^2_-(c), sigma^2_+(c) 요거 추정

이 때 사용하는 h를 silverman rule을 이용하여 제공함

제공된 h를 이용하여 emphirical distribution of X, 분산 추정치 사용함



point : 얘네들이 다 consistent하다! + 다른 consistent한 estimator 사용해도 된다.



2. second derivative 추정

이 때 사용하는 h는 third order polynomial regression 이용하여 fit (local linear 사용하기 전에 RD design에서 사용했던 방법 중 하나)

왜 추정하냐? h 추정할 때 third derivative가 필요하기 때문 

추정치 바탕으로 h2+, h2- 추정하고, second order local polynomial regression second derivative 추정



3. 다 넣어서 h_opt 구하기





좋았다 + regularization안한거보다 한게 더 좋음



만약에 적용을 한다



수식적인 접근 : local likelihood 식이랑 위 논문에서 제공된 식이랑 다름

problem : 밑의 증명이 local linear regressor가 closed form이어서 증명이 가능했는데, 내가 사용할 모형은 closed form이 아닐 거 같아서 생각을 좀 더 해야 함 + categorical outcome 해석을 더 해야함  - 흐름은 이해했는데, 이 부분 해결을 못함 - 요거 어떻게 풀어냈는지 좀 알아내야 함 



+ 찾은 논문 : 그 논문 + ordinal outcome에 대해서 같은 방법론 적용한거 밖에 못찾음

찾는 방법 google scholar - 개많음

"" - 너무 적음 - 맞나?



scopus, 다른 사이트 두개에서 찾았는데 안보였음







