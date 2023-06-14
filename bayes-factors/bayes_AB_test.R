# Code applying tools from the "Bayesian Inference for the A/B Test" 2020 Hoffmann masters thesis paper
# Goes through two approaches: the odds-ratio approach and the common approach. Scroll down for inputs and results of the latter
# I prefer the odds-ratio approach since it fulfills more needs such as finding evidence of no effect

# -------------------------
# Install or Load packages
# -------------------------

#install.packages('abtest')
#install.packages('bayesAB')
library(abtest)
library(bayesAB)


# -------------------------
# Data & Expectations
# -------------------------
# Example used below: A/B test for a website redesign
# Control Group: 35 users out of 100 total visitors convert
# Test Group: 50 users out of 100 total visitors convert
# Based on baselines, you expect that the Test Group will have between a 5% and 20% lift over the Control Group

n1 <- 100 # population in Control Group
y1 <- 35 # number of successes in Control Group
n2 <- 100 # population in Test Group
y2 <- 50 # number of successes in Test Group


# -------------------------
# The "Odds Ratio Approach":
# Created to fulfill 3 criteria: evidence can be obtained in favor of null hypothesis, can be monitored as data accumulates, prior expert knowledge can be taken into account
# -------------------------

# Inputs:
rangeLow <- 0.05 # minimum benefit percentage you expect one of the conditions to have over the other
rangeHigh <- 0.2 # maximum benefit percentage you expect one of the conditions to have over the other
nSamples = 10000 # number of samples to run in the simulation, the more the smoother the posterior distribution curves; 10000 is the default
priorProb <- c(0, 0.25, 0.25, 0.5) # prior probability of hypotheses H1, H+, H- and H0, respectively; c(0, 0.25, 0.25, 0.5) is the default
names(priorProb) <- c("H1", "H+", "H-", "H0") # change order if you changed the order of them in the definition of priorProb above
# H1: Two-sided alternative hypothesis stating success probabilities are not equal (doesn't specify which one)
# H+: One-sided alternative hypothesis stating Test Group success probability is larger than Control Group
# H-: One-sided alternative hypothesis stating Control Group success probability is larger than Test Group
# H0: Null hypothesis stating success probabilities are equal (no difference between Test Group and Control Group)


# Run Test:
rangeAvg <- (rangeLow + rangeHigh) / 2
priorPar <- elicit_prior(q = c(rangeLow, rangeAvg, rangeHigh) # expectation range of lift for the Test Group over the Control Group
                      , prob = c(.025, .5, .975) # probabilities corresponding to quantiles above assuming a 95% uncertainty interval (e.g. median is 0.5)
                      , what = "arisk" # absolute risk: difference of success probability in the two conditions
                      , hypothesis = "H+") # quantiles provided for H+ hypothesis: Test > Control

# NOTE: if you have no idea of baselines or do NOT want to use past knowledge, use the below for a very uninformative prior instead
#priorPar <- elicit_prior(q = c(-1.95, 0, 1.95) # uniform assumption
#                          , prob = c(.025, .5, .975) # probabilities corresponding to quantiles above assuming a 95% uncertainty interval (e.g. median is 0.5)
#                          , what = "logor") # logor: log odds ratio

set.seed(1) # for repeating sampling
data <- data.frame(y1, n1, y2, n2) # create data frame of population and successes for Condition A (1: Control) and Condition B (2: Test)
abTest <- ab_test(data = data
                  , prior_par = priorPar # list with prior parameters for distributions
                  , prior_prob = priorProb # list with prior probabilities on the four hypotheses
                  , nsamples = nSamples)

interpret_BF <- function(BF) {
  if (BF < 1) {
    return('No Evidence')
  }
  else if (BF >= 1 & BF < 3) {
    return('Anecdotal Evidence')
  }
  else if (BF >= 3 & BF < 10) {
    return('Moderate Evidence')
  }
  else if (BF >= 10 & BF < 30) {
    return('Strong Evidence')
  }
  else if (BF >= 30 & BF < 100) {
    return('Very Strong Evidence')
  }
  else if (BF >= 100) {
    return('Decisive Evidence')
  }
}

# Interpretation:
print(abTest)

# Posterior probabilities show how data has increased/decreased plausibility of each of the above hypotheses

# Bayes Factors:
# BF10: evidence toward H1 (difference exists between conditions) over H0 (no difference between conditions)
# BF01 = 1 / BF10: evidence toward H0 (no difference between conditions) over H1 (difference exists between conditions)
# BF+0: evidence toward H+ (Test Group success probability is larger than Control Group) over H0 (no difference between conditions)
# BF-0: evidence toward H- (Control Group success probability is larger than Test Group) over H0 (no difference between conditions)

paste('Likelihood in favor of Test Group > Control Group over no effect between groups: ', round(abTest$bf$bfplus0,2), ' (', interpret_BF(abTest$bf$bfplus0), ')', sep="")
paste('Likelihood in favor of Control Group > Test Group over no effect between groups: ', round(abTest$bf$bfminus0,2), ' (', interpret_BF(abTest$bf$bfminus0), ')', sep="")
paste('Likelihood in favor of no effect over any effect between groups: ', round(1/abTest$bf$bf10,2), ' (', interpret_BF(1/abTest$bf$bf10), ')', sep="")


# -------------------------
# The "Common Approach":
# Assumptions: two success probabilities are independent, an effect is always present! (if you want to find evidence of no effect, use above method instead)
# -------------------------

# Inputs:
nSamples = 100000 # number of samples to run in the simulation, the more the smoother the posterior distribution curves


# Test:
control <- c(rep(1, y1), rep(0, n1 - y1)) # create vector of Control Group successes (1) and failures (0)
test <- c(rep(1, y2), rep(0, n2 - y2)) # create vector of Test Group successes (1) and failures (0)
abTestSimple <- bayesTest(A_data = test,
                     B_data = control,
                     priors = c('alpha' = 1, 'beta' = 1), # uninformative parameters for the distribution. Beta(1,1) = Uniform(0,1)
                     n_samples = nSamples,
                     distribution = 'bernoulli')

summary(abTestSimple)
# result P(A > B) value means we can be that % certain that version A is better than version B
#plot(abTestSimple)
