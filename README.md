# A/B Testing Tools
---
### Sample sizing and experiment result calculator 

Made using Streamlit and available here: https://ab-testing-calculator.streamlit.app/

I've worked on a lot of experimentation at a few companies and product groups often ask the data team to tell them how many users they need in order to see a behavior change as a reuslt of a new or adjusted product feature. The challenge is that the answer always depends on the situation - what's the base rate of the thing they want to measure and how risky they want to be or how expensive it is to run the experiment etc. 

There are a number of good online calculators out there too but some are too technical and I wanted to make one that our specific audience understands for their use cases as well as try out [Python Streamlit](https://streamlit.io/). 

Calculator features:
- sample sizing for your experiment (how many users do I need for my test?)
- results calculation (was my experiment successful?) for discrete/binary metrics (e.g. did a user login?)

Future features:
- continuous metrics (e.g. total money spent, number of actions taken etc.) rather than binary ones
- Bayesian A/B testing with Bayes factors rather than frequentist (this will be based on the Bayes Factor R code referenced below and will convert into python)

---
### Bayes Factors

The sub-folder here describes a project on building an A/B testing calculator that uses Bayesian methods rather than frequentist ones:
https://github.com/idanileiko/ab-testing-tools/tree/main/bayes-factors

Bayesian A/B testing offers the following benefits over frequentist A/B testing methods:
- The ability to "peek" at the results of an experiment in progress
- Bayes factors show the degree to which one hypothesis is more likely than another instead of a binary significant or not result
- Can find evidence for the null hypothesis whereas traditional methods can't

The downside is the results are more complicated to interpret, especially for non-technical stakeholders who are more accustomed to a significant-or-not value. Additionally, Bayesian testing can be heavily influenced by prior knowledge especially in the cases of little or uninformative data.
