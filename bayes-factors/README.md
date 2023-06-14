## A/B Testing using Bayes Factors

The R code shows an example Bayesian test using an R package from one of the resources cited at the end of this document [^1]. Most of the code is covered in some of the papers but I added comments that explain all the parameters and test interpretation for that specific use case. Another resource describes why you would want to use Bayesian A/B testing [^2].

> As a caution, in the case of small sample sizes and insufficient data, keep in mind it'll be more heavily weighted toward the priors but it'll still provide at least some directional trend when looking at the Bayes factors.

There are two methods that run the test in the R code:
1. an odds-ratio approach
2. a method called a "common" approach
   
I prefer the odds-ratio approach for getting the full view of all the Bayes factors but the latter offers a much more simple approach if you're sure that an effect is present and just want to know the certainty level.

Because this methodology can find evidence toward multiple hypotheses instead of just rejecting one null hypothesis, it's more complex than frequentist significance testing. You have to be careful about interpreting the results and look at several factors to understand the whole picture. All the Bayes factors show likelihoods of one hypothesis relative to the other. There are four hypotheses going on in the A/B test case:
* **H1**: the success probability differs between the control and the experimental condition but does not specify _which one_ is higher
* **H0**: the success probability between the conditions is identical
* **H+**: the success probability in the experimental condition is _higher_ than in the control condition
* **H-**: the success probability in the experimental condition is _lower_ than in the control condition

Subsequently, the three Bayes factors that the R function outputs are:
* **BF10**: likelihood of H1 _relative to_ H0; BF01 is not output by default but is simply 1 / BF10
* **BF+0**: likelihood of H+ _relative to_ H0
* **BF-0**: likelihood of H- _relative to_ H0

The interpretation of the BF value themselves can be fairly arbitrary [^3] but generally a BF > 10 is considered strong evidence. For Bayesian approaches, it often comes down to levels of uncertainty rather than decisive binary answers.

> An important part of the R code is the posteriors are heavily influenced by the priors you assign if there's insufficient data. So if you start playing around with the data inputs and don't know/want to have the expert knowledge baselines, switch the prior definition from lines 49-52 over to the currently commented-out one in lines 54-57.

### Future work:

Currently the R code only looks at conversions at the end of the test but in theory and as the paper describes, this methodology allows you to continuously run the code on a time series/sequential data set even as you're still collecting data. This is the ideal use case I'd want to use it for so that (for example) we can cut an expensive experiment early if it's not showing results).

Future work involves writing that sequential code as well as code for continuous variables (such as the amount of money spent by users) instead of binary conversion rates.


[^1]: [Bayesian Inference for the A/B Test: Example Applications with R and JASP](https://www.researchgate.net/publication/352307826_Bayesian_Inference_for_the_AB_Test_Example_Applications_with_R_and_JASP)
[^2]: [Informed Bayesian Inference for the A/B Test](https://www.researchgate.net/publication/356659768_Informed_Bayesian_Inference_for_the_AB_Test)
[^3]: [Bayes Factor interpretation](https://en.wikipedia.org/wiki/Bayes_factor#Interpretation)
