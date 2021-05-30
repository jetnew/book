Causal Inference
======

Causal inference is the inference of the effect of any treatment of $T$ on the outcome $Y$, based on the causal structure of the underlying process, e.g. inferring the effect of a treatment on a disease.

## Correlation and Causation

Correlation is not causation. A confounder variable $Z$ can confound the association between the treatment variable $T$ and outcome variable $Y$. The potential outcome $Y(t)$ is the outcome of taking a particular treatment $T=t$.

# The Fundamental Problem of Causal Inference

It is impossible to observe all potential outcomes for a given individual, because the opposite treatment cannot be repeated on the same individual in the past. Counterfactuals are potential outcomes that are not observed and cannot be observed because they are counter to fact. Because counterfactuals cannot be observed, it is fundamentally a missing data problem.

## Individual and Average Treatment Effects

The individual treatment effect (ITE) $\tau_i$ is the causal effect of taking a treatment on a potential outcome. The ITE computes the difference in potential outcomes of an individual given different treatments.

$$ \tau_i \overset{\Delta}{=} Y_i(1)-Y_i(0)$$

The average treatment effect (ATE) $\tau$ is an average over ITEs, because the ITE cannot easily be accessed. The ATE computes the difference in expected potential outcomes given different treatments.

$$ \tau \overset{\Delta}{=} E[Y_i(1)-Y_i(0)] = E[Y(1)-Y(0)] $$

## Associational and Causal Quantities

The ITE and ATE are causal quantities that are not equal to the conditional expectation, which is an associational quantity, due to the existence of confounders.

When confounding exists, the ATE:

$$ \underbrace{E[Y(1)]-E[Y(0)]}_{causal} \neq \underbrace{E[Y\vert T=1]-E[Y\vert T=0]}_{associational}$$

When confounding does not exist, the ATE:

$$ \underbrace{E[Y(1)]-E[Y(0)]}_{causal} = \underbrace{E[Y\vert T=1]-E[Y\vert T=0]}_{associational} $$

Identification is the computation of causal quantities from associational quantities, which requires assumptions about the causal structures of the underlying processes.

## Randomized Control Trials

A randomized control trial (RCT) is the random assignment of individuals into the treatment or control group. As a result, the treatment variable $T$ does not have any causal parents, removing the confounding association, isolating the causal association to allow identification.

# Assumptions for Identification

## Identifiability

A causal quantity is identifiable if it can be computed from a purely statistical/associational quantity.

## Ignorability / Exchangeability

$$ (Y(1),Y(0)) \perp\!\!\!\perp T $$

Ignorability of how the treatment was assigned, i.e. assuming random assignment of the treatment, allows reduction of the ATE to the associational quantity.

$$ E[Y(1)-E[Y(0)] $$

$$ = E[Y(1)\vert Y=1]-E[Y(0)\vert T=0] \tag{Ignorability} $$

$$ = E[Y\vert T=1]-E[Y\vert T=0] $$

Exchangeability of treatment and control groups means that the same outcomes are observed should they be exchanged and thus are comparable. In reality, ignorability/exchangeability does not usually hold due to confounding but can be applied using randomized control trials (RCTs).

## Assumption 1: Conditional Exchangeability / Unconfoundedness

$$ (Y(1),Y(0)) \perp\!\!\!\perp T\vert X $$

Controlling for $X$ by conditioning on $X$ results in treatment groups being comparable, removing any non-causal association between treatment $T$ and outcome $Y$.

$$ E[Y(1)-Y(0)\vert X] $$

$$ = E[Y(1)\vert X] - E[Y(0)\vert X] $$

$$ = E[Y(1)\vert T=1,X] - E[Y(0)\vert T=0,X] $$

$$ = E[Y\vert T=1,X] - E[Y\vert T=0,X] $$

The marginal effect before assuming unconditional exchangeability can be obtained by marginalizing out $X$.

$$ E[Y(1)-Y(0)] $$

$$ = E_X E[Y(1)-Y(0)\vert X] $$

$$ = E_X [E[Y\vert T=1,X] - E[Y\vert T=0,X]] $$

In contrast to exchangeability, conditional exchangeability allows conditioning on X to identify the causal effect.

## Assumption 2: Positivity / Overlap / Common Support

For all values of covariates $x$ present in the population of interest (such that $P(X=x)>0$),

$$ 0 < P(T=1\vert X=x) < 1 $$

If positivity is violated, a zero probability event will be conditioned on, translating into a division by zero when Baye's rule is applied on:

$$ E_X [E[Y\vert T=1,X] - E[Y\vert T=0,X]] $$

A positivity violation happens when, within some subgroup of data, every individual receives the treatment or every individual receives the control, and the causal effect of treatment vs. control cannot be estimated because the alternative is not observed.

## Assumption 3: No Interference

No interference holds if one individual's outcome is unaffected by another's treatment, i.e. one individual's outcome is only a function of one's own treatment.

$$ Y_i(t_1, ..., t_i, ..., t_n) = Y_i(t_i) $$

## Assumption 4: Consistency

Consistency of the treatment holds if the observed outcome $Y$ is actually the potential outcome under the observed treatment $T$.

$$ (T=t \rightarrow Y=Y(t)) $$

$$ \equiv Y=Y(T) $$

A consistency violation happens if the treatment specification is too coarse. Different versions of the same treatment exists, resulting in different potential outcomes even if the same treatment is applied.

# Identification of the Causal Effect

When all 4 assumptions (unconfoundedness, positivity, no interference, consistency) hold, the causal effect can be identified from the associational quantities.

$$E[Y(1)-Y(0)] \tag{No Interference}$$

$$=E[Y(1)]-E[Y(0)] \tag{Linearity of Expectation}$$

$$=E_X[E[Y(1)|x]-E[Y(0)|X]] \tag{Law of Iterated Expectations}$$

$$=E_X[E|Y(1)|T=1,X]-E[Y(0)|T=0,X]] \tag{Unconfoundedness and Positivity}$$

$$=E_X[E[Y|T=1,X]-E[Y|T=0,X]] \tag{Consistency}$$

# References

Introduction to Causal Inference from a Machine Learning Perspective, [Brady Neal 2020](https://www.bradyneal.com/causal-inference-course).
