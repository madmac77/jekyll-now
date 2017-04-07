# Project 3: Housing Price Modelling

## Introduction

You are a data science consultant for a real-estate firm in Ames, Iowa. They have tasked you with building a model to predict home prices. But, there's a catch: although currently they have 82 features in their database, they want to save costs and record less data in the future. So, they want you to build a model using only the below subset of features:

- Lot.Area
- Utilities
- Neighborhood
- Bldg.Type
- House.Style
- Overall.Qual
- Overall.Cond
- Year.Built
- Year.Remod.Add
- Roof.Style
- Roof.Matl
- Gr.Liv.Area
- Full.Bath
- Half.Bath
- Bedroom.AbvGr
- Kitchen.AbvGr
- Mo.Sold
- Yr.Sold
- SalePrice


They are also very focused on location and want to know 1) where the most sales are taking place, 2) where the most expensive houses are located, and 3) if this is changing over time.  

**So your task is:**
- to build them a model to predict house prices
- to provide them with an analysis of neighborhoods to answer their question
- to provide them with any other insights from the data that you think are important (what other relationships exist?)


---

###  Requirements

Using the provided dataset, create a linear regression model and an executive summary write-up based on the project outline.

Your work must:
**Identify the problem**
- Write a high quality problem statement
- Describe the goals of your study and criteria for success

**Acquire the data**
- Obtain the data [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (_you will need to sign up to Kaggle to see the data_)
- Use only the "train" data set
- Remember: you are only allowed to use a subset of the data features, as specified above

**Explore the data**
- Import data using Pandas
- Perform exploratory analysis methods with visualization and statistical analysis
- State the risks and assumptions of your data

**Mine the data**
- Create necessary derived columns from the data
- Format, clean, slice, and combine the data in Python

**Refine the data**
- Determine outliers, skew distribution of important variables (if any)
- Determine correlations in the data
- Validate findings using statistical analysis (p-values, confidence intervals) as applicable

**Build a data model**
- Complete linear regressions using scikit-learn or statsmodels and interpret your findings
- Calculate and plot predicted probabilities and/or present tables of results
- Evaluate model fit by using loss functions, including mean absolute error, mean squared error, and root mean squared error, or r-squared

**Present the results**
- Create a Jupyter writeup that provides a dataset overview with visualizations, statistical analysis, data cleaning methodologies, and models
- Create a writeup on the interpretation of findings including an executive summary with conclusions and next steps. This can be included within your notebook or as a blog post.


***Bonus!:***

 - Recommend additional data that might improve your models


### Necessary Deliverables / Submission

- Materials must be in a clearly labeled Jupyter notebook that satisfies project requirements:
- Materials must be submitted to Github and posted as an Issue on the DSI-5 GHE Project 3 page.


--
### Useful Resources
- [Documentation for SKLearn](http://scikit-learn.org/stable/user_guide.html)
- [What is regularization?](https://www.quora.com/What-is-regularization-in-machine-learning)

---