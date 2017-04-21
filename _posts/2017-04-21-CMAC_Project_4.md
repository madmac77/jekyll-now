
### Description

This week we're learning about logistic regressions and web scraping. Let's put these skills to the test!

You're working as a data scientist for a contracting firm that's rapidly expanding. Now that they have their most valuable employee (you - heyo!), they need to leverage data to win more contracts. Your firm offers technology and scientific solutions and wants to be competitive in the hiring market. Your principal thinks the best way to gauge salary amounts is to take a look at what industry factors influence the pay scale for these professionals.

Aggregators like [Indeed.com](https://www.indeed.com) regularly pool job postings from a variety of markets and industries. Your job is to understand what factors most directly impact data science salaries and effectively and accurately find appropriate data science-related jobs in your metro region.

#### Project Summary

In this project, we will practice two major skills. Collecting data by scraping a website and then building a binary predictor with Logistic Regression.

We are going to collect salary information on data science jobs in a variety of markets. Then using the location, title, and summary of the job, we will attempt to predict a corresponding salary for that job. While most listings DO NOT come with salary information (as you will see in this exercise), being to able extrapolate or predict the expected salaries for other listings will be extremely useful for negotiations :).

Normally we could use regression for this task; however, instead we will convert this into a classification problem and use Logistic Regression.

- **Question**: Why would we want this to be a classification problem?
- **Answer**: While more precision may be better, there is a fair amount of natural variance in job salaries; therefore, predicting a range (e.g. high or low) may be useful.

The first part of assignment will be focused on scraping data, and the second will be focused on using the listings with salary information to build a model and predict salaries.

Your job is to:

1. Collect data on data science salary trends from a job listings aggregator for your analysis.
  - Select and parse data from at least ~1000 postings for jobs, potentially from multiple location searches.
2. Find out what factors most directly impact salaries (title, location, department, etc.). In this case, we do not want to predict mean salary, as would be done in a regression. Your boss believes that salary is better represented in categories (i.e., "high" and "low") than continuously.
  - Test, validate, and describe your models. What factors predict salary category? How do your models perform?
3. Prepare a presentation for your Principal detailing your analysis.

**BONUS PROBLEMS:**
1. Your boss would rather tell a client incorrectly that they would get a lower salary job than tell a client incorrectly that they would get a high salary job. Adjust one of your logistic regression models to ease her mind, and explain what it is doing and any tradeoffs. Plot the ROC curve.
2. Text variables and regularization:
  - **Part 1**: Job descriptions contain more potentially useful information you could leverage. Use the job summary to find words you think would be important and add them as predictors to a model.
  - **Part 2**: Gridsearch parameters for Ridge and Lasso for this model and report the best model.


**Goal:** Scrape & clean data, run logistic regression, derive insights, present findings.

---

### Requirements

- Scrape and prepare your data using BeautifulSoup.
- A team Jupyter Notebook with your regression analysis for a peer audience of data scientists.
- An individual blog post describing your findings, with two sections: the first for a non-technical audience, and the second for data scientist peers.



Scraping job listings from Indeed.com
We will be scraping job listings from Indeed.com using BeautifulSoup. Luckily, Indeed.com is a simple text page where we can easily find relevant entries.
First, look at the source of an Indeed.com page: (http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l=New+York&start=10")
Notice, each job listing is underneath a div tag with a class name of result. We can use BeautifulSoup to extract those.
Setup a request (using requests) to the URL below. Use BeautifulSoup to parse the page and extract all results (HINT: Look for div tags with class name result)

## IMPORT RELEVANT LIBRARIES AND MODULES


```python
import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
```

## DEFINED FORMUALAE:


```python
def fix_salaries(salary_string):
    salary = 0
    salary_list = salary_string.replace("$","").replace(",","").strip().split()
    if "-" in salary_list[0]:
        temp = salary_list[0].split("-")
        salary = sum([float(a) for a in temp])/len(temp)
    else:
        salary = float(salary_list[0])
    if salary_list[-1] == "month":
        #salary *12 months
        salary *= 12
    elif salary_list[-1] == "hour":
        #40hrs/week, 50 weeks
        salary *= 2000
    return salary
```


```python
def extract_location_from_job_result(result):
    a = result.find('span',class_="location")
    return None if (a == None) else a.text.strip()

def extract_company_from_job_result(result):
    a = result.find('span',class_="company")
    return None if (a == None) else a.text.strip()

def extract_jobtitle_from_job_result(result):
    a = result.find('a', {"data-tn-element":"jobTitle"})
    return None if (a == None) else a.text.strip()
    
def extract_salary_from_job_result(result):
    a = result.find('nobr', text=True)
    return None if (a == None) else a.text.strip()
```


```python
def location_fixer(location_string):
    location_list = location_string.strip().split(',')
    location= location_list[0]
    if len(location_list)>1:
        state= location_list[1][1:3]
        location= str(location_list[0])+", "+str(state)
    return location
```

## URL/ WEBSCRAPING
* New York City
* 2000 listings
* Extracting 'company', 'location', 'salary', 'title'


```python
url_template = "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l={}&start={}"
max_results_per_city = 2000

results = []

for city in set(['New+York']):
    for start in range(0, max_results_per_city, 10):
        # Grab the results from the request (as above)
        r = requests.get(url_template.format(city,start))
        soup= BeautifulSoup(r.content, "lxml")
        result= soup.findAll('div', class_='result')
        for i in result:
            results.append((extract_location_from_job_result(i), extract_company_from_job_result(i), \
          extract_salary_from_job_result(i), extract_jobtitle_from_job_result(i)))
print results
```

    [(u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Associate - Machine Learning'), (u'New York, NY', u'Twitter', None, u'Machine Learning'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Data Analyst - Research, Insights & Segmentation'), (u'New York, NY', u'Plated', None, u'Data Engineer'), (u'New York, NY', u'Wolters Kluwer', None, u'Junior Data Scientist'), (u'New York, NY', u'FXcompared', None, u'Data Scientists'), (u'New York, NY', u'AIG', None, u'Data Science Analyst'), (u'New York, NY', u'NYU School of Medicine', None, u'Senior Data Analyst'), (u'New York, NY', u'Microsoft', None, u'Principal Data Scientist - NY or Redmond'), (u'New York, NY', u'Aetion', None, u'Data Engineer'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Schrodinger', None, u'Machine Learning Infrastructure Engineer'), (u'New York, NY', u'Freestar', None, u'Machine Learning Engineer'), (u'New York, NY', u'Aetion', None, u'Data Engineer'), (u'New York, NY 10017 (Midtown area)', u'Pfizer', None, u'Senior Scientist, Oncology Target Discovery'), (u'New York, NY', u'Hudson Data', None, u'Sr. Data Scientist'), (u'New York, NY', u'Columbia University', None, u'Associate Research Scientist'), (u'New York, NY', u'Venturi Ltd', u'$200,000 a year', u'Data Scientist ( FinTech / Python / R / Machine Learning / B...'), (u'New York, NY', u'NYU School of Medicine', None, u'Bioinformatics Programmer'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Senior Data Scientist - Banking Analytics'), (u'New York, NY', u'SKIP', None, u'Machine Learning/Deep Learning Engineer'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Advanced Imaging Data Analyst'), (u'New York, NY', u'Research Foundation of The City University of New...', u'$36,476 - $47,478 a year', u'Research associate'), (u'New York, NY', u'New York Life Insurance Co', None, u'Corporate Vice President, Lead Data Scientist (46330)'), (u'New York, NY', u'Capital One', None, u'Machine Learning Data Engineer'), (u'New York, NY', u'NYU School of Medicine', None, u'Research Data Associate'), (u'New York, NY', u'NewYork-Presbyterian/Midtown Offices', None, u'Statistical Data Analyst'), (u'New York, NY 10005 (Financial District area)', u'WeWork', None, u'People Analytics Lead'), (u'New York, NY', u'E*TRADE FINANCIAL', None, u'Senior Decision Scientist, Campaign Analytics, Decision Mana...'), (u'New York, NY', u'Foursquare', None, u'Analytics - Ads Team'), (u'New York, NY', u'JPMorgan Chase', None, u'Digital Intelligence - Data Engineer'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'HiringCatalyst', None, u'Data Engineer - Streaming Video Data Analytics - Apache Stor...'), (u'New York, NY', u'Institute for the Study of Decision Making', None, u'Junior Research Scientist (Rolling Admission)'), (u'New York, NY 10011 (Chelsea area)', u'Scripps Networks', None, u'Associate Analyst Digital Research'), (u'New York, NY 10016 (Gramercy area)', u'pymetrics', None, u'Junior Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'Aetna', None, u'Lead Data Scientist'), (u'New York, NY', u'Hospital for Special Surgery', None, u'Statistical Analyst'), (u'New York, NY', u'Bonobos', None, u'Senior Analyst - Business Intelligence'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Data Scientist, Analytics'), (u'New York, NY 10016 (Gramercy area)', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10003 (Greenwich Village area)', u'Medidata Solutions', None, u'Statistical Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003 (Greenwich Village area)', u'J.Crew Group, Inc.', None, u'Sr. Data Scientist - Test & Learn Analytics'), (u'New York, NY', u'Fractal Industries', None, u'Data Analytics Engineer'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Quantitative Research Associate - Full-Time Campus Hire'), (u'New York, NY 10007 (Financial District area)', u'Conde Nast', None, u'Analyst, Data Access & Advanced Analysis'), (u'New York, NY 10017 (Midtown area)', u'Pfizer Inc.', None, u'Clinical Data Scientist (Associate, entry level with DM expe...'), (u'New York, NY', u'JPMorgan Chase', None, u'Lead Data Scientist, Digital Intelligence'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'McKinsey Solutions- Healthcare Analytics and Delivery Manage...'), (u'New York, NY 10003 (Greenwich Village area)', u'J.Crew Group, Inc.', None, u'Sr. Data Scientist - Mixed Media Modeling'), (u'New York, NY', u'Weight Watchers International', None, u'Data Engineer'), (u'New York, NY', u'WebMD', None, u'Associate Statistician'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10018 (Clinton area)', u'Comcast', None, u'Data Scientist Intern'), (u'New York, NY', u'Hearst Magazines', None, u'Custom Research Analyst'), (u'New York, NY', u'AdTheorent', None, u'Machine Learning Engineer'), (u'New York, NY', u'Dailymotion', None, u'Data Engineer'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Product Manager - Healthcare Analytics, McKinsey New Venture...'), (u'New York, NY', u'NYU School of Medicine', None, u'Data Analyst'), (u'New York, NY', u'AIG', None, u'Senior Business Analyst, Underwriting Analytics'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Institutional Data Analyst'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Data Coordinator II - Transplant Institute - Full Time - Day...'), (u'New York, NY', u'New York Life Insurance Co', None, u'SENIOR ASSOCIATE, Data Scientist (New York, New York)'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Columbia University', None, u'Data Analyst'), (u'New York, NY', u'NYU School of Medicine', None, u'Data Analyst'), (u'New York, NY', u'FULLBEAUTY Brands', None, u'Statistical/ Database Marketing Analyst'), (u'New York, NY 10032 (Washington Heights area)', u'Morgan Stanley', None, u'Machine Learning Engineer - NY'), (u'New York, NY', u'Yahoo! Inc.', None, u'Research Scientist'), (u'New York, NY', u'BuzzFeed', None, u'Data Analyst'), (u'New York, NY', u'NBCUniversal', None, u'Sr. Data Scientist'), (u'New York, NY', u'WebMD', None, u'Associate Statistician'), (u'New York, NY', u'Facebook', None, u'Software Engineer, Machine Learning'), (u'New York, NY', u'Solomon Page', None, u'Freelance Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'DEPT OF HEALTH/MENTAL HYGIENE', u'$59,708 - $72,246 a year', u'Data Analyst, Bureau of Immunization'), (u'New Hyde Park, NY 11040', u'Northwell Health', None, u'Data Research Analyst - Office of Chief Nurse Executive'), (u'New York, NY', u'United Nations', None, u'Statistician'), (u'New York, NY', u'Research Foundation of The City University of New...', u'$80,000 - $87,000 a year', u'Senior Research Analyst'), (u'New York, NY 10012 (Little Italy area)', u'Neustar, Inc.', None, u'Data Strategy Analyst'), (u'New York, NY', u'POLICE DEPARTMENT', u'$42,288 - $63,519 a year', u'Statistician, Level I'), (u'New York, NY 10017 (Midtown area)', u'Access Staffing LLC', u'$180,000 a year', u'Quantitative Analyst'), (u'Manhattan, NY', u'DEPARTMENT OF FINANCE', u'$70,286 - $80,829 a year', u'Data Analyst/Modeler'), (u'New York, NY', u'New York State Office of the Attorney General', None, u'Research / Data Analyst'), (u'New York, NY 10022 (Midtown area)', u'Two Harbors Investment Corp.', None, u'Quantitative Risk Analyst, Two Harbors'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'JPMorgan Chase', None, u'Machine Learning Engineer (Associate) - Intelligent Solution...'), (u'New York, NY', u'E*TRADE FINANCIAL', None, u'Sr. Market Research Analyst'), (u'New York, NY', u'Capital One', None, u'Data Scientist - Manager'), (u'New York, NY 10003 (Greenwich Village area)', u'J.Crew Group, Inc.', None, u'Sr. Data Scientist - Customer Glidepaths & Audience Analytic...'), (u'New York, NY', u'NYU School of Medicine', None, u'Scientific Programmer'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Analyst - Healthcare Analytics & Delivery, McKinsey New Vent...'), (u'Jersey City, NJ', u'EXL', None, u'Machine Learning Scientist/Senior Scientist'), (u'New York, NY', u'Life Time Fitness', None, u'Ultimate Hoops Statistician'), (u'New York, NY', u'Solomon Page', None, u'Freelance Data Scientist'), (u'New York, NY', u'Schrodinger', None, u'Machine Learning Researcher'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'NYU Langone Health System', None, u'Management Analyst (per diem)'), (u'New York, NY', u'JPMorgan Chase', None, u'Machine Learning Engineer (Associate) - Intelligent Solution...'), (u'New York, NY', u'HiringCatalyst', None, u'Lead Data Engineer - Streaming Video Analytics - Apache Stor...'), (u'New York, NY', u'Custoria', None, u'Customer Success Analyst'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Analyst - Healthcare Analytics & Delivery, McKinsey New Vent...'), (None, None, None, None), (u'New York, NY', u'AdTheorent', None, u'Director Data Science'), (u'New York, NY', u'Solomon Page', None, u'Freelance Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'Aetna', None, u'Sr Data Scientist'), (u'New York, NY 10031 (Hamilton Heights area)', u'Urban Scholars Program, City College of New York', u'$40 an hour', u'Computer Science (Data Analysis) Instructor'), (u'New York, NY', u'Facebook', None, u'Instagram - Software Engineer, Machine Learning'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'Manhattan, NY', u"ADMIN FOR CHILDREN'S SVCS", u'$70,286 - $88,213 a year', u'Data Analyst'), (u'New York, NY', u'Consol Partners LLC', None, u'Senior Data Scientist - NYC - Competitive Salary!'), (u'New York, NY', u'Maru/EDR', None, u'Senior Research Operations Analyst - Maru/edr'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Machine Learning Infrastructure Lead'), (u'New York, NY 10031 (Hamilton Heights area)', u'Urban Scholars Program, City College of New York', u'$40 an hour', u'Computer Science (Data Analysis) Instructor'), (u'New York, NY 10017 (Midtown area)', u'Analytic Recruiting', u'$160,000 a year', u'Senior Data Scientist'), (u'New York, NY', u'NYU Langone Health System', None, u'Management Analyst (per diem)'), (u'New York, NY', u'Custoria', None, u'Customer Success Analyst'), (u'Clark, NJ 07066', u"L'Oreal USA", None, u'Intern/Coop - Consumer Hair Evaluation'), (u'New York, NY', u'AdTheorent', None, u'Director Data Science'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'Jersey City, NJ', u'EXL', None, u'Deep Learning Scientist/Senior Scientist'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Developer \u2013 Banking Analytics'), (u'Manhattan, NY', u"ADMIN FOR CHILDREN'S SVCS", u'$70,286 - $88,213 a year', u'Data Analyst'), (u'New York, NY', u'Consol Partners LLC', None, u'Senior Data Scientist - NYC - Competitive Salary!'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY 10001 (Chelsea area)', u'Phoenix Marketing International', None, u'Senior Research Analyst'), (u'Jersey City, NJ', u'Verisk Analytics', None, u'Lead Data Scientist'), (u'New York, NY 10012 (Little Italy area)', u'Meetup', None, u'Strategist'), (u'New York, NY', u'Citi', None, u'Rates Quantitative Analyst \u2013 Associate/VP'), (u'New York, NY 10261 (Murray Hill area)', u'MassMutual Financial Group', None, u'Data Visualization Engineer'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'iSpot.tv, Inc.', None, u'Senior Research Analyst'), (u'New York, NY', u'Yahoo! Inc.', None, u'Sr. Research Scientist'), (u'New York, NY 10002 (Lower East Side area)', u'MSCI Inc.', None, u'Quantitative and data analyst'), (u'New York, NY 10017 (Midtown area)', u'Analytic Recruiting', u'$160,000 a year', u'Senior Data Scientist'), (u'New York, NY', u'DigitasLBi', None, u'Vice President/Director, Data & Analysis'), (u'New York, NY 10014 (West Village area)', u'SAP', None, u'ETL - Data Scientist - Hadoop Developer'), (u'New York, NY', u'Soci\xe9t\xe9 G\xe9n\xe9rale', None, u'Quantitative Model Validation Analyst'), (u'New York, NY 10013 (Tribeca area)', u'Galvanize', None, u'Lead Instructor, Principal Data Scientist'), (u'New York, NY', u'Enterprise Select', u'$180,000 a year', u'Sr Quantitative Finance Analyst'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'LABORATORY SUPERVISOR'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10001 (Chelsea area)', u'Phoenix Marketing International', None, u'Senior Research Analyst'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Developer \u2013 Banking Analytics'), (u'New York, NY', u'Soci\xe9t\xe9 G\xe9n\xe9rale', None, u'Junior Research Analyst - Index'), (u'New York, NY', u'DEPT OF HEALTH/MENTAL HYGIENE', u'$70,286 - $80,829 a year', u'Data Manager \u2013 Researcher, Bureau of Children, Youth & Famil...'), (u'New York, NY 10018 (Clinton area)', u'Quartet', None, u'Data Platform Engineer'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Data Analyst II - AC Adolescent Health - Full Time - Days'), (u'New York, NY 10018 (Clinton area)', u'JW Player', None, u'Junior Data Engineer'), (u'New York, NY 10003 (Greenwich Village area)', u'SoundCloud', None, u'Senior Data Scientist, Analytics'), (u'New York, NY', u'Dailymotion', None, u'Senior Data Scientist, Ad-Tech'), (u'New York, NY 10018 (Clinton area)', u'JW Player', None, u'Data Engineer'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Dailymotion', None, u'Product Manager, (Data)'), (u'New York, NY', u'ELOQUII', None, u'Senior Software Developer'), (u'New York, NY', u'NBCUniversal', None, u'Principal, Data Science, Audience Studio'), (u'New York, NY', u'Viacom', None, u'FULL STACK ENGINEER, DATA VISUALIZATION & ARCHITECTURE'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Data Analyst II - AC Adolescent Health - Full Time - Days'), (u'New York, NY 10019 (Midtown area)', u'Merkle Inc.', None, u'Analytics Director'), (u'New York, NY 10012 (Little Italy area)', u'Neustar, Inc.', None, u'Senior Data Strategy Analyst'), (u'New York, NY', u'Capital One', None, u'Senior Associate, Quantitative Analyst \u2013 Market Risk'), (u'New York, NY', u'Axius Technologies', None, u'Sr. Data Scientist'), (u'New York, NY 10018 (Clinton area)', u'JW Player', None, u'Junior Data Engineer'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'LABORATORY SUPERVISOR'), (u'Jersey City, NJ', u'Forbes Media LLC', None, u'Senior Data Scientist - Contract to Hire'), (u'New York, NY', u'Enterprise Select', u'$180,000 a year', u'Sr Quantitative Finance Analyst'), (u'New York, NY', u'Kennedy Unlimited Inc, Professional Staffing', u'$130,000 - $140,000 a year', u'Predictive Analytics (Machine Learning)'), (u'New York, NY', u'Darwin Recruitment', u'$120 - $140 an hour', u'Machine Learning Engineer - NLP - Java - Python - New York'), (u'New York, NY', u'UBS', None, u'Quantitative Analyst'), (u'New York, NY 10005 (Financial District area)', u'Celmatix', None, u'Research Associate'), (u'New York, NY', u'Icahn School of Medicine at Mount Sinai', None, u'Institute Postdoctoral Fellow (Omics & Big Data)'), (u'New York, NY', u'Harnham', u'$180,000 a year', u'Senior Data Scientist - NLP and Machine Learning'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Senior Scientist, Department of Psychiatry'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Enterprise Solutions Quant Data Product Specialist'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'ASSOCIATE RESEARCHER II - FULL TIME - DAYS - MSH'), (u'Parsippany, NJ', u'Wyndham Destination Network', None, u'Senior Data Scientist, Analytics'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Business Intelligence Analyst'), (u'New York, NY 10011 (Chelsea area)', u'Enterprise Select', u'$140,000 a year', u'Senior Data Engineer'), (u'New York, NY 10018 (Clinton area)', u'JW Player', None, u'Data Engineer'), (u'New York, NY 10007 (Financial District area)', u"Moody's Investors Service", None, u'Quantitative Analyst'), (u'New York, NY 10022 (Midtown area)', u'Credit Suisse', None, u'Quantitative Analyst \u2013 Counterparty Credit Risk (VP)'), (u'New York, NY 10011 (Chelsea area)', u'eBay Inc.', None, u'Data Engineer'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR SCIENTIST'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10011 (Chelsea area)', u'eBay Inc.', None, u'Data Engineer'), (u'New York, NY 10016 (Gramercy area)', u'The Forum Group', None, u'Data Analyst (Statistical/Multivariate)'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR SCIENTIST'), (u'New York, NY 10038 (Financial District area)', u'Enterprise Select', u'$130,000 a year', u'Data Scientist'), (u'New York, NY 10002 (Lower East Side area)', u'MSCI Inc.', None, u'Senior Associate, Quantitative Analyst'), (u'New York, NY', u'NBA', None, u'Manager, Data Management'), (u'New York, NY', u'Reed Business Information', None, u'Senior Web Analyst'), (u'New York, NY', u'Kennedy Unlimited Inc, Professional Staffing', u'$130,000 - $140,000 a year', u'Predictive Analytics (Machine Learning)'), (u'New York, NY', u'Harnham', u'$180,000 a year', u'Senior Data Scientist - NLP and Machine Learning'), (u'New York, NY', u'Citi', None, u'Equities Quantitative Analyst'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'Jersey City, NJ', u'ISO', None, u'Sr. Lead Data Scientist'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR SCIENTIST'), (u'Union Beach, NJ', u'IFF', None, u'Research Scientist, Automation and Signal Processing'), (u'New York, NY', u'ELOQUII', None, u'Senior Software Developer'), (u'New York, NY', u'AIG', None, u'Senior Science Analyst'), (u'East Hanover, NJ 07936', u'Mondelez International', None, u'Senior Scientist I'), (u'Berkeley Heights, NJ 07922', u'Aequor Technologies', u'$70 an hour', u'Statistical Programmer/ Statistician Medical Affairs'), (u'New York, NY', u'Kennedy Unlimited Inc, Professional Staffing', u'$130,000 - $140,000 a year', u'Predictive Analytics (Machine Learning)'), (u'New York, NY 10002 (Lower East Side area)', u'MSCI Inc.', None, u'Senior Associate, Quantitative Analyst'), (u'New York, NY', u'Bloomberg', None, u'User Experience Designer - Data Distribution and Governance'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Darwin Recruitment', u'$120 - $140 an hour', u'Machine Learning Engineer - NLP - Java - Python - New York'), (u'New York, NY', u'Harnham', u'$180,000 a year', u'Senior Data Scientist - NLP and Machine Learning'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Data Management and Education Assistant'), (u'New York, NY', u'The Shipyard', None, u'Director of Machine Learning & Real Time Bidding Systems'), (u'New York, NY 10012 (Little Italy area)', u'L2', None, u'Junior Data Scientist'), (None, None, None, None), (u'New York, NY 10003 (Greenwich Village area)', u'Integral Ad Science', None, u'Director, Data Science'), (u'New York, NY', u'Soci\xe9t\xe9 G\xe9n\xe9rale', None, u'Quantitative Analyst - Regulatory Models'), (u'New York, NY 10003 (Greenwich Village area)', u'Integral Ad Science', None, u'Data Engineer, Data Science'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'ASSOCIATE RESEARCHER II - FULL TIME - DAYS - MSH'), (u'New York, NY', u'Yahoo! Inc.', None, u'Sr Research Scientist- Yahoo Video'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Harnham', u'$150,000 a year', u'Senior Data Scientist - Modeling'), (u'New York, NY 10007 (Financial District area)', u"Moody's Investors Service", None, u'VP-Senior Research Analyst'), (u'Union Beach, NJ', u'IFF', None, u'Research Scientist, Automation and Signal Processing'), (u'New York, NY 10032 (Washington Heights area)', u'Morgan Stanley', None, u'Lead Data Scientist: HR Business Intelligence & Advanced Ana...'), (u'New York, NY 10018 (Clinton area)', u'Quartet', None, u'Data Scientist - Platform Development'), (u'New York, NY', u'Business Insider, Inc.', None, u'Senior Research Analyst, E-Commerce'), (u'New York, NY 10016 (Gramercy area)', u'The Forum Group', None, u'Data Analyst (Statistical/Multivariate)'), (u'New York, NY', u'GUTTMACHER', None, u'Senior Research Scientist'), (u'New York, NY 10006 (Financial District area)', u'Essex Lake Group', None, u'Modeler'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Data Management and Education Assistant'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10011 (Chelsea area)', u'Enterprise Select', u'$140,000 a year', u'Senior Data Engineer'), (u'New York, NY', u'7Park Data', None, u'Research Analyst - Consumer'), (u'New York, NY 10013 (Tribeca area)', u'Blue Apron', None, u'Data Ops Engineer'), (u'New York, NY', u'Capital One', None, u'Data Analysis Manager'), (u'New York, NY', u'Bonobos', None, u'Director of Insights and Analytics'), (u'New York, NY 10014 (West Village area)', u'Delos', None, u'Building Environment Research Scientist'), (u'New York, NY', u'Harnham', u'$150,000 a year', u'Data Engineer'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Senior Data Scientist- Legal'), (u'New York, NY', u'Research Foundation of The City University of New...', None, u'Computational Science Assistant (Multiple Positions)'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003 (Greenwich Village area)', u'J.Crew Group, Inc.', None, u'Director - Analytics'), (u'New York, NY', u'Business Insider, Inc.', None, u'Senior Research Analyst, E-Commerce'), (u'New York, NY', u'GUTTMACHER', None, u'Senior Research Scientist'), (u'Berkeley Heights, NJ', u'inVentiv Health Clinical', None, u'Statistician'), (u'New York, NY', u'Brooklyn Data Science', None, u'Senior Data Science Instructor at NYC Data Science Academy'), (u'New York, NY', u'TED', None, u'Business Intelligence Analyst - TED Mobile + Platforms'), (u'New York, NY', u'Integrated Management Resources, LLC', None, u'Quantitative Analyst-Loan Level Financial Data-SQL'), (u'New York, NY', u'W2O Group', None, u'Associate Research Analyst (Marketing and Social Media)'), (u'New York, NY 10007 (Financial District area)', u"Moody's Investors Service", None, u'VP-Senior Research Analyst'), (u'New York, NY', u'Helix', None, u'Senior Data Analyst'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10013 (Tribeca area)', u'Blue Apron', None, u'Data Ops Engineer'), (u'New York, NY', u'Yahoo! Inc.', None, u'Sr Research Scientist- Yahoo Video'), (u'New York, NY', u'TED', None, u'Business Intelligence Analyst - TED Mobile + Platforms'), (u'New York, NY', u'Reed Business Information', None, u'Customer Consultant - PURE'), (u'Manhattan, NY', u'DEPT OF HEALTH/MENTAL HYGIENE', u'$59,708 - $64,485 a year', u'Environmental Analyst'), (u'New York, NY', u'adMarketplace', None, u'Data Scientist'), (u'New York, NY', u'Brooklyn Data Science', None, u'Data Scientist at CompStak'), (u'New York, NY', u'NYU School of Medicine', None, u'Staff Andrologist-IVF'), (u'New York, NY 10014 (West Village area)', u'Delos', None, u'Building Environment Research Scientist'), (u'New York, NY', u'S.C. International', u'$130,000 a year', u'Sr. Modeler/Data Scientist - Life-7660'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'SFL Scientific', None, u'Data scientist, machine vision'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Senior Data Scientist- Legal'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'Columbia University', None, u'Associate Data & Research Analyst'), (u'New York, NY', u'Capital One', None, u'Senior Machine Learning Engineer'), (u'New York, NY', u'Columbia University', None, u'Postdoctoral Research Scientist'), (u'New York, NY 10018 (Clinton area)', u'Princeton Consulting', u'$130,000 - $160,000 a year', u'Quantitative Engineer & Analyst'), (u'Berkeley Heights, NJ', u'inVentiv Health Clinical', None, u'Statistician'), (u'Clark, NJ 07066', u"L'Oreal USA", None, u'Creative Qualitative - Sr Scientist-Evaluation'), (u'New York, NY 10018 (Clinton area)', u'Quartet', None, u'Data Scientist - Health Data Science'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Rockstar New York', None, u'Senior Data Engineer - BI & Analytics'), (u'New York, NY', u'NYU Langone Health System', None, u'Sr. Management Analyst'), (u'New York, NY', u'Reed Business Information', None, u'Customer Consultant - PURE'), (u'New York, NY', u'AXA', None, u'Chief Data and Analytics Officer'), (u'New York, NY 10013 (Tribeca area)', u'CKM Advisors', u'$15 - $20 an hour', u'Part Time Office Manager'), (u'New York, NY', u'Columbia University', None, u'Senior Bioinformatician'), (u'New York, NY', u'General Assembly', None, u'Lead Data Science Instructor'), (u'Clark, NJ 07066', u"L'Oreal USA", None, u'Creative Qualitative - Sr Scientist-Evaluation'), (u'New York, NY', u'Averity', u'$150,000 - $190,000 a year', u'Principal Data Scientist ($190k)'), (u'New York, NY 10012 (Little Italy area)', u'Oscar Insurance', None, u'Risk Operations Associate'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Helix', None, u'Senior Data Analyst'), (u'New York, NY 10012 (Little Italy area)', u'Oscar Insurance', None, u'Risk Operations Associate'), (u'New York, NY', u'605', u'$95,000 - $130,000 a year', u'Frontend Engineer'), (u'New York, NY', u'NYU School of Medicine', None, u'In House Temp Interview Day (April 26, 2017)'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR BIOSTATISTICIAN'), (u'New York, NY 10003 (Greenwich Village area)', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'VROOM', u'$140,000 - $180,000 a year', u'Data Team Leader (VP, Director)'), (u'New York, NY', u'Socure', None, u'Senior Data Scientist'), (u'New York, NY', u"Harry's", None, u'Senior Sensory Scientist'), (u'New York, NY 10001 (Chelsea area)', u'Remedy Partners Inc.', None, u'Data Science Lead'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10013 (Tribeca area)', u'CKM Advisors', u'$15 - $20 an hour', u'Part Time Office Manager'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Tumor Registrar - Temporary/Per Diem'), (u'New York, NY 10002 (Lower East Side area)', u'Spectrum', None, u'Research Analyst - Media'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Research Study Specialist'), (u'New York, NY', u'1199SEIU Family of Funds', None, u'Research Analyst/Accountant III'), (u'New York, NY', u'NYU School of Medicine', None, u'Simulation Operations Specialist'), (u'New York, NY 10014 (West Village area)', u'Delos', None, u'Research Scientist, Health Economics and Outcomes'), (u'Union Beach, NJ', u'International Flavors and Fragrances', None, u'Research Scientist, Automation and Signal Processing'), (u'New York, NY', u'New York State Office of the Attorney General', None, u'Financial Analyst'), (u'New York, NY', u'Tumblr', None, u'Sr. Data Scientist - User & Product Analytics'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'New York State Office of the Attorney General', None, u'Financial Analyst'), (u'New York, NY', u'Tumblr', None, u'Sr. Data Scientist - User & Product Analytics'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Coding Liasion - Clin Doc Excellence'), (u'New York, NY', u'HBO', None, u'HBO Senior Data Engineer'), (u'New York, NY', u'Elenion Technologies', None, u'Senior Photonic Design Engineer'), (u'Bronx, NY', u'Albert Einstein College of Medicine', None, u'Research Technician A'), (u'New York, NY', u'Graviton Consulting Services,Inc', None, u'Data Scientist'), (u'New York, NY', u'PDT Partners', None, u'C++/Python Engineer'), (u'New York, NY 10016 (Gramercy area)', u'FactSet Research Systems', None, u'Data Science Intern'), (u'New York, NY 10018 (Clinton area)', u'RapidSOS', None, u'People Operations Analyst'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'NBA', None, u'Data Scientist'), (u'New York, NY', u'SFL Scientific', None, u'Data scientist, machine vision'), (u'New York, NY', u'Bloomberg', None, u'Data Acquisition Strategy Leader'), (u'New York, NY 10017 (Midtown area)', u'Wunderman', None, u'Senior Analyst'), (u'New York, NY 10018 (Clinton area)', u'Quartet', None, u'Data Scientist - Health Data Science'), (u'New York, NY', u'Dailymotion', None, u'Director of Data Science, Ad-Tech'), (u'New York, NY', u'Columbia University', None, u'Postdoctoral Research Scientist'), (u'New York, NY', u'Data Inc.', None, u'Python/Quantitative Finance Analyst'), (u'New York, NY 10002 (Lower East Side area)', u'Spectrum', None, u'Research Analyst - Media'), (u'New York, NY 10014 (West Village area)', u'Squarespace', None, u'Software Engineer - Data'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Quantitative Software Engineer, Feature Modeling Analytics'), (u'New York, NY', u'NewOak Capital LLC', None, u'Quantitative Analyst'), (u'New York, NY', u'New York State Office of the Attorney General', None, u'Financial Analyst'), (u'New York, NY', u'BuzzFeed', None, u'Associate Creative'), (u'Berkeley Heights, NJ', u'Aequor Technologies', None, u'Statistician'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Data Research and Acceleration Analyst'), (u'New York, NY', u'Socure', None, u'Senior Data Scientist'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Analyst - Pharmaceutical and Medical Products, McKinsey Solu...'), (u'New York, NY', u'Kelton Global', None, u'Senior Analyst, Quantitative Research'), (u'New York, NY', u'1199SEIU Family of Funds', None, u'Research Analyst/Accountant III'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'Clark, NJ 07066', u"L'Oreal USA", None, u'Co-op Intern - Cosmetic Evaluation'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Financial Analyst I'), (u'New York, NY', u'DTS', None, u'Machine Learning Software Engineer'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Quantitative Software Engineer, Feature Modeling Analytics'), (u'New York, NY', u'OppenheimerFunds', None, u'Director, Marketing Advanced Analytics & Campaign Management'), (u'New York, NY', u'Reed Business Information', None, u'Manager, Marketing Database & Technology'), (u'New York, NY', u'Hiring Catalyst, LLC', None, u'Full Stack Engineer - Customer Service Powered by NLP'), (u'New York, NY', u'Bloomberg', None, u'Data Acquisition Strategy Leader'), (u'New York, NY', u'DTS', None, u'Machine Learning Engineer (VP) Morgan Stanley'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Quantitative Researcher in Machine Learning'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u"Moody's Analytics", None, u'Director-Sr Research Analyst'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR BIOSTATISTICIAN'), (u'New York, NY 10003 (Greenwich Village area)', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'JLL', None, u'Analyst, Research'), (u'New York, NY', u'Eli Lilly', None, u'Research Associate-FDE'), (u'New York, NY', u'Harmony Institute', None, u'Researcher'), (u'New York, NY', u'Graviton Consulting Services,Inc', None, u'Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Coding Liasion - Clin Doc Excellence'), (u'New York, NY', u'Twitter', None, u'Software Engineer'), (u'New York, NY', u'Elenion Technologies', None, u'Senior Photonic Design Engineer'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', None, None, u'Director, Analytics and Insights'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Senior Programmer/Analyst'), (u'New York, NY', u'The Economist Group', None, u'Research Analyst Intern'), (u'New York, NY', u'Casper', None, u'Senior Analyst, Site Optimization and Analytics'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Data Scientist / Researcher'), (u'New York, NY 10038 (Financial District area)', u'Guttmacher Institute', None, u'Senior Research Scientist'), (u'New York, NY', u'DTS', None, u'Machine Learning Software Engineer'), (u'New York, NY', u'POLICE DEPARTMENT', u'$70,286 - $88,213 a year', u'Data Analyst'), (u'New York, NY', u'Hospital for Special Surgery', None, u'Research Data and Statistical Analyst'), (u'New York, NY 10017 (Midtown area)', u'Analytic Recruiting', None, u'Data Analytics Engineer Record Linkage Modeler- CRE Firm'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'BNY Mellon', None, u'Sr Quantitative Rate Risk Analyst (QRM)'), (u'New York, NY', u'Hiring Catalyst, LLC', None, u'Full Stack Engineer - Customer Service Powered by NLP'), (u'New York, NY', u'Data Inc.', None, u'Python/Quantitative Finance Analyst'), (u'Manhattan, NY', u'HRA/DEPT OF SOCIAL SERVICES', u'$78,630 - $94,500 a year', u'Senior Project Manager'), (u'New York, NY', u'NYU Langone Health System', None, u'Project Manager'), (u'New York, NY', u'Worldgroup Careers', None, u'2017 Analytics Summer Internship, Momentum'), (u'New York, NY', u'Foursquare', None, u'Senior Data Engineer'), (u'New York, NY', u'Reed Business Information', None, u'Manager, Marketing Database & Technology'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Financial Analyst I'), (u'New York, NY', u'Rockstar New York', None, u'Analytics Project Manager'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'Berkeley Heights, NJ', u'Aequor Technologies', None, u'Statistician'), (u'New York, NY', u'Big Cloud', None, u'Senior Data Scientist - Applied NLP'), (u'New York, NY', u'Shutterstock', None, u'Algorithm Engineer'), (u'New York, NY', u'Mount Sinai Genetic Testing Laboratory, Icahn Scho...', None, u'Bioinformatician II'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Research Administration Specialist'), (u'New York, NY', u'NewOak Capital LLC', None, u'Quantitative Analyst'), (u'New York, NY', u'NYU School of Medicine', None, u'Senior Grants Specialist'), (u'New York, NY 10019 (Midtown area)', u'Merkle Inc.', None, u'Senior UI Developer'), (u'New York, NY', None, None, u'Director, Analytics and Insights'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Integr Hlth Spec-Mass Therapy'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'DTS', None, u'Machine Learning Software Engineer'), (u'New York, NY 10002 (Lower East Side area)', u'PricewaterhouseCoopers LLC', None, u'Halo for Financial Services Data Science Senior Associate'), (u'New York, NY', u'HarperCollins Publishers Inc.', None, u'Data Science Manager'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Bioinformatics Analyst'), (u'New York, NY', u'JLL', None, u'Analyst, Research'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SR ASSOCIATE RESEARCHER'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Biostatistician I - Health Policy'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Senior Performance Analyst (Dashboard Analytics - Hospital O...'), (u'New York, NY', u'DTS', None, u'Machine Learning Engineer (VP) Morgan Stanley'), (u'New York, NY', u'JPMorgan Chase', None, u'Paid Social and Search, Marketing Analytics, VP - NY, NY'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'JPMorgan Chase', None, u'Customer Graph Digital Data Scientist'), (u'New York, NY', u'Yodlee', None, u'Equity Research Analyst'), (u'New York, NY', u'Ambulatory/Outpatient NYU Hospitals Center', None, u'Cardiac Sonographer'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Clinical Research Monitor'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Accenture Analytics-Data Science Consultant'), (u'Bronx Zoo, NY', u'Wildlife Conservation Society', None, u'Molecular Pathology Post-Doctoral Fellow'), (u'New York, NY', u'Venturi Ltd', u'$120,000 - $200,000 a year', u'Senior Data Scientist ( FinTech / Python / R / ML / Big Data...'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Research Administration Specialist'), (u'New York, NY', u'Capital One', None, u'Senior Quantitative Analyst'), (u'New York, NY', u'Capital One', None, u'Senior Analyst, Commercial Strategy & Innovation'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Data Science Product Manager'), (u'New York, NY', u'FSAStore.com, Inc.', None, u'Business Intelligence & Consumer Insights Director'), (u'New York, NY', u'Harmony Institute', None, u'Researcher'), (u'New York, NY', u'ektello', None, u'Senior Data Scientist'), (u'New York, NY', u'Averity', u'$150,000 - $250,000 a year', u'Director of Data Science, Analytics'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Research Specialist'), (u'New York, NY', u'Schrodinger', None, u'Scientific Software Developer, Protein-Ligand Database'), (u'New York, NY', u'Capital One', None, u'Data Scientist'), (u'New York, NY', u'Schrodinger', None, u'Senior Scientist, Machine Learning and Virtual Screening'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Citi', None, u'Equities MQA - Quantitative Analyst-VP - Global Cash Trading...'), (u'New York, NY', u'JPMorgan Chase', None, u'Big Data Product Strategy & Development Manager - Intelligen...'), (u'Manhattan, NY', u'The Bachrach Group', u'$15 an hour', u'Data Analysis Assistant (Temporary)'), (u'New York, NY', u'Columbia University', None, u'Associate Director of Development'), (u'New York, NY', u'Open Systems Technologies, Inc.', None, u'Research Scientist'), (u'Summit, NJ 07901', u'McKinsey & Company', None, u'Marketing Analytics & Effectiveness Specialist'), (u'New York, NY', u'UJA Federation of New York', None, u'Data Analysis and Reporting Manager'), (u'New York, NY 10003 (Greenwich Village area)', u'Medidata Solutions', None, u'Senior Business Analyst'), (u'New York, NY', u'Schrodinger', None, u'Senior Scientist, Cryo-EM Model Fitting'), (u'Manhattan, NY', u'HRA/DEPT OF SOCIAL SERVICES', u'$70,286 - $83,000 a year', u'Project Manager'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Bioinformatics Analyst'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Biostatistician I - Health Policy'), (u'Manhasset, NY', u'Siemens', None, u'Healthcare Decision Scientist'), (u'Jersey City, NJ', u'Verisk Insurance Solutions', None, u'Lead Data Scientist'), (u'New York, NY 10019 (Midtown area)', u'Merkle Inc.', None, u'Senior UI Developer'), (u'New York, NY', u'The Consortium Inc.', None, u'Data, Scala, Spark, Python, SQL, risk analysis, financial mo...'), (u'New York, NY', u'NYU School of Medicine', None, u'Asst Dir-Applied BioinformLabs'), (u'New York, NY', u'Viacom', None, u'MANAGER, ANALYTICS & INSIGHTS'), (u'New York, NY 10003 (Greenwich Village area)', u'Medidata Solutions', None, u'Senior Business Analyst'), (u'New York, NY', u'IJC Associates', None, u'Machine Learning Researcher'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Citi', None, u'Equities MQA - Quantitative Analyst-VP - Global Cash Trading...'), (u'New York, NY', u'Hopper', None, u'Software Engineer'), (u'New York, NY', u'NYU School of Medicine', None, u'Senior Grants Specialist'), (u'New York, NY', u'NBCUniversal', None, u'Senior Analyst, Digital & Cross-Platform Ad Sales Research'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'RESEARCH ASSISTANT III'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'ektello', None, u'Senior Data Scientist'), (u'New York, NY', u'JPMorgan Chase', None, u'Paid Social and Search, Marketing Analytics, VP - NY, NY'), (u'New York, NY', u'HarperCollins Publishers Inc.', None, u'Data Science Manager'), (u'New York, NY', u'Capital One', None, u'Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'VICE US', None, u'Head of Business Intelligence'), (u'New York, NY', u'Weill Cornell Medical College', None, u'IT Business Analyst I'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Data Science Manager - Premium Analytics'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Integr Hlth Spec-Mass Therapy'), (u'New York, NY', u'Schrodinger', None, u'Senior Scientist, Machine Learning and Virtual Screening'), (u'Manhattan, NY', u'HRA/DEPT OF SOCIAL SERVICES', u'$70,286 - $83,000 a year', u'Project Manager'), (u'New York, NY', u'Schrodinger', None, u'Scientific Software Developer, Protein-Ligand Database'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Quantitative Analyst'), (u'New York, NY', u'Soci\xe9t\xe9 G\xe9n\xe9rale', None, u'Quantitative Analyst - Model Validation'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Senior Performance Analyst (Dashboard Analytics - Hospital O...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017 (Midtown area)', u'Wunderman', None, u'Global Client Lead'), (u'New York, NY 10018 (Clinton area)', u'Harris Allied', None, u'Java Developer and Machine learning, Contract'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Accenture Analytics - Data Science Analyst'), (u'Manhattan, NY', u'DEPARTMENT OF TRANSPORTATION', u'$60,189 - $83,440 a year', u'Bike Share Data Scientist'), (u'Jersey City, NJ', u'Verisk Insurance Solutions', None, u'Lead Data Scientist'), (u'New York, NY', u'Viacom', None, u'MANAGER, ANALYTICS & INSIGHTS'), (u'New York, NY', u'NYU Langone Health System', None, u'Enterprise Data Governance Analyst'), (u'New York, NY', u'TransReach', None, u'Quantitative Finance Analyst'), (u'New York, NY 10017 (Midtown area)', u'Wunderman', None, u'Associate Project Manager'), (u'Bronx, NY', u'Albert Einstein College of Medicine', None, u'Research Fellow'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10018 (Clinton area)', u'Quartet', None, u'Community Specialist'), (u'Jersey City, NJ 07306 (Journal Square area)', u'BNY Mellon', None, u'Senior Developer - Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Per Diem Cardiac Sonographer ( Nights and Weekends)'), (u'New York, NY', u'DISH Network', None, u'General Manager, Data and Analytics'), (u'New York, NY', u'Newsela', None, u'Director of Data Science'), (u'Jersey City, NJ', u'EXL', None, u'Analytics Product Manager'), (u'New York, NY', u'NYU School of Medicine', None, u'ParentCorps Educator'), (u'New York, NY', u'DEPT OF ENVIRONMENT PROTECTION', u'$70,286 - $88,213 a year', u'City Research Scientist'), (u'New York, NY', u'New York Life Insurance Co', None, u'NYL Post Grad Internship: Data Scientist/Statistician'), (u'New York, NY', u'Dstillery', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10002 (Lower East Side area)', u'Bluecore', None, u'Software Engineer'), (u'New York, NY 10013 (Tribeca area)', u'Momentum', None, u'2017 Analytics Summer Internship, Momentum'), (u'New York, NY 10012 (Little Italy area)', u'Neustar, Inc.', None, u'Advisory Services Consultant'), (u'New York, NY 10169 (East Harlem area)', u'Clarion Partners', None, u'Research Analyst/Associate'), (u'Manhattan, NY', u'DEPT OF HEALTH/MENTAL HYGIENE', u'$59,708 - $72,246 a year', u'Panel Maintenance Analyst, World Trade Center Health Registr...'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Director - Clinical Informatics & Medical Economics.'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Accenture Analytics-Data Science Consultant'), (u'Kenilworth, NJ', u'On-Board Services', None, u'Pharmaceutical Process Engineer'), (u'East Hanover, NJ', u'Whiz Finder Corporation', u'$105,000 - $125,000 a year', u'Senior Statistical Programmer'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Data Science Manager - Premium Analytics'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'East Hanover, NJ', u'Whiz Finder Corporation', u'$105,000 - $125,000 a year', u'Senior Statistical Programmer'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Data Science Manager - Premium Analytics'), (u'New York, NY', u'Brooklyn Data Science', None, u'Data Scientist (NLP) at Privco'), (u'New York, NY 10022 (Midtown area)', u'Crowe Horwath', None, u'Data Analysis Manager'), (u'New York, NY', u'Shutterstock', None, u'Senior Machine Learning/Computer Vision Engineer'), (u'New York, NY', u'Weill Cornell Medical College', None, u'IT Business Analyst I'), (u'New York, NY 10005 (Financial District area)', u'Parsons Corporation', None, u'Sr. Technical Consultant - Insitu Technologies SME'), (u'New York, NY 10017 (Midtown area)', u'Mount Sinai Health System', None, u'TECHNOLOGY SPECIALIST II'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'International Patient Relations Representative'), (u'New York, NY 10261 (Murray Hill area)', u'MassMutual Financial Group', None, u'Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'Manhattan, NY', u'DEPARTMENT OF TRANSPORTATION', u'$60,189 - $83,440 a year', u'Bike Share Data Scientist'), (u'Manhattan, NY', u'DEPT OF HEALTH/MENTAL HYGIENE', u'$59,708 - $72,246 a year', u'Panel Maintenance Analyst, World Trade Center Health Registr...'), (u'Clark, NJ 07066', u"L'Oreal USA", None, u'Scientist or Senior Scientist \u2013 Instrumental Evaluation'), (u'New York, NY', u'NYU Langone Health System', None, u'Revenue Integrity Analyst'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Director - Clinical Informatics & Medical Economics.'), (u'New York, NY', u'Dstillery', None, u'Senior Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'Simulmedia', None, u'Data Scientist'), (u'New York, NY', u'NYU School of Medicine', None, u'ParentCorps Educator'), (u'New York, NY', u'Attune Insurance Services', None, u'Front End Engineer'), (u'Summit, NJ 07901', u'McKinsey & Company', None, u'Delivery Manager - Periscope, McKinsey Solutions'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Products Analytics Consultant'), (u'New York, NY 10018 (Clinton area)', u'JW Player', None, u'Data Scientist'), (u'New York, NY 10005 (Financial District area)', u'Axelon Services Corporation', None, u'Statistical Data Analyst'), (u'New York, NY 10038 (Financial District area)', u'Birchbox', None, u'Director, Data Science'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Diagnostic Sonographer'), (u'New York, NY 10038 (Financial District area)', u'Guttmacher Institute', None, u'Director of International Research'), (None, None, None, None), (u'New York, NY', u'Weill Cornell Medical College', None, u'Medical Records Coordinator'), (u'Manhasset, NY', u'Siemens', None, u'Healthcare Decision Scientist'), (u'New York, NY', u'Liberty Environmental, Inc.', None, u'Environmental Scientist'), (u'New York, NY 10065 (Upper East Side area)', u'Memorial Sloan Kettering Cancer Center', None, u'Associate Manager Computational Biology - Cancer Genomics'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Shutterstock', None, u'Senior Machine Learning/Computer Vision Engineer'), (u'New York, NY', u'Feedzai', None, u'Software Engineer'), (u'New York, NY', u'Brooklyn Data Science', None, u'Data Scientist (NLP) at Privco'), (u'New York, NY 10038 (Financial District area)', u'Guttmacher Institute', None, u'Director of International Research'), (u'New York, NY', u'NYU Langone Health System', None, u'Enterprise Data Governance Analyst'), (u'New York, NY', u'PMES', u'$155,000 a year', u'Sr Data Scientist - NLP & CODING REQUIRED'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'International Patient Relations Representative'), (u'New York, NY 10014 (West Village area)', u'UncommonGoods', None, u'Director of Analytics'), (u'New York, NY', u'Oliver James Associates', u'$160,000 a year', u'Data Scientist'), (u'Bronx, NY', u'Albert Einstein College of Medicine', None, u'Bioinformatics Analyst'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'Manhattan, NY', u'DEPT. OF HOMELESS SERVICES', u'$70,286 - $88,213 a year', u'Quality Management Coordinator'), (u'New York, NY', u'NYU Langone Health System', None, u'Sr. Analyst-Privacy & Compliance'), (u'Manhattan, NY', u'DEPT OF PARKS & RECREATION', u'$70,286 - $81,000 a year', u'Environmental Protection Project Manager'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Data Science Pharma Manager'), (u'New York, NY 10022 (Midtown area)', u'TV One', None, u'Sr. Research Analyst'), (u'New York, NY', u'Citi', None, u'Agency Lending Quantitative Analyst'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Analytics Strategy and Transformation Manager - Utilities (N...'), (u'New York, NY', u'Ambulatory/Outpatient NYU Hospitals Center', None, u'Phlebotomist'), (u'New York, NY 10012 (Little Italy area)', u'Oscar Insurance', None, u'Associate Actuary'), (u'New York, NY 10005 (Financial District area)', u'WeWork', None, u'Senior Software Engineer (Machine Learning)'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'West Orange, NJ 07052', u'Kessler Foundation', None, u'Neuroimaging Quantitative Analyst'), (u'New York, NY', u'All-In Analytics', u'$140,000 - $150,000 a year', u'Senior Data Scientist'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Manager - Cognitive Data Scientist Natural Language Processi...'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Sr Associate Researcher, Liver Department'), (u'New York, NY', u'Mirador Real Estate, LLC', u'$45,000 a year', u'Listing Coordinator'), (u'New York, NY 10065 (Upper East Side area)', u'Memorial Sloan Kettering Cancer Center', None, u'Associate Manager Computational Biology - Cancer Genomics'), (u'New York, NY 10005 (Financial District area)', u'Parsons Corporation', None, u'Sr. Technical Consultant - Insitu Technologies SME'), (u'New York, NY', u'T. Rowe Price', None, u'T. Rowe Price - NYC Technology Development Center'), (u'Whippany, NJ', u'Oxford Solutions', None, u'Lead Statistical Analyst - Oncology'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Quality Assurance Specialist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10018 (Clinton area)', u'JW Player', None, u'Data Scientist'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'RN Clinical Data Coordinator - Cardiology'), (u'New York, NY', u'BuzzFeed for Video Internship/Fellowship/Residency', None, u'Video Fellow - Tasty'), (u'Bronx, NY', u'Albert Einstein College of Medicine', None, u'Research Technician B'), (u'New York, NY 10005 (Financial District area)', u'WeWork', None, u'Senior Software Engineer (Machine Learning)'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'Averity', u'$50 - $100 an hour', u'Quantitative Analyst at Leading Analytic Driven Hedge Fund (...'), (u'New York, NY', u'Icon plc', None, u'Health Economist'), (u'Jersey City, NJ', u'EXL', None, u'Healthcare PowerBI Developer, Decision Analytics Services'), (u'New York, NY', u'New York City Dept. of Environmental Protection', None, u'Projects Controls Specialist- Water Quality Unit'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Jobspring Partners', u'$120,000 - $150,000 a year', u'Senior Data Scientist'), (u'New York, NY 10005 (Financial District area)', u'WeWork', None, u'Senior Software Engineer (Machine Learning)'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'Icon plc', None, u'Health Economist'), (u'Jersey City, NJ', u'EXL', None, u'Healthcare PowerBI Developer, Decision Analytics Services'), (u'New York, NY', u'New York City Dept. of Environmental Protection', None, u'Projects Controls Specialist- Water Quality Unit'), (u'New York, NY', u'Flatiron Health', None, u'Senior Software Engineer, Machine Learning'), (u'New York, NY', u'Icahn School of Medicine at Mount Sinai', u'$55,000 a year', u'Postdoctoral Fellow'), (u'New York, NY', u'General Assembly', None, u'Part Time Data Science Instructor'), (u'New York, NY 10001 (Chelsea area)', u'Kognito', None, u'Project Manager for Game-Based Simulations'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'Bronx, NY', u'Albert Einstein College of Medicine', None, u'Research Technician B'), (u'Oakland, NJ', u'Topcon', None, u'Software Engineer (Machine Learning)'), (u'New York, NY', u'Icon plc', None, u'Health Economist'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Client Development Manager - PriceMetrix, McKinsey Solutions'), (u'New York, NY', u'Elenion Technologies', None, u'Senior Optoelectronics Packaging Engineer'), (u'New York, NY', u'Smith Hanley Associates', u'$75,000 - $90,000 a year', u'Data Scientist'), (u'New York, NY', u'Deutsche Bank', None, u'Lead Software Engineer'), (u'New York, NY', u'Ambulatory/Outpatient NYU School of Medicine', None, u'FGP Sleep Technologist 1- Temporary'), (u'New York, NY 10017 (Midtown area)', u'Wunderman', None, u'Presentation Specialist'), (u'New York, NY', u'General Assembly', None, u'Part Time Data Science Instructor'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Drexel University', u'$2,400 - $3,200 a month', u'Green Infrastructure Field Technician'), (u'New York, NY', u'Yahoo! Inc.', None, u'Sales Data Insights Mgr'), (u'Carnegie Hill, NY', u'T3 Trading Group LLC', None, u'Statistical Pairs Trading Position'), (u'New York, NY', u'Aetion', None, u'Front End Engineer'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'Healthcare Operations Consultant, Payors'), (u'New York, NY 10012 (Little Italy area)', u'Oscar Insurance', None, u'Associate Actuary'), (u'New York, NY', u'Wolters Kluwer', None, u'Data Scientist'), (u'New York, NY', u'HarperCollins Publishers Inc.', None, u'Data Scientist'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Systems Analyst'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Cognitive Data Scientist Natural Language Processing'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'BNY Mellon', None, u'Manager Research Analyst II - Wealth Management'), (u'Carnegie Hill, NY', u'T3 Trading Group LLC', None, u'Statistical Pairs Trading Position'), (u'New York, NY', u'Icahn School of Medicine at Mount Sinai', u'$55,000 a year', u'Postdoctoral Fellow'), (u'New York, NY', u'Innovations for Poverty Action', None, u'Senior Research & Data Analyst'), (u'New York, NY', u'NYU Langone Health System', None, u'Associate Director, Population Health IT Solutions and Strat...'), (u'New York, NY', u'BuzzFeed for Video Internship/Fellowship/Residency', None, u'Video Fellow - Tasty'), (u'Jersey City, NJ', u'EXL', None, u'Healthcare PowerBI Developer, Decision Analytics Services'), (u'New York, NY', u'NYU School of Medicine', None, u'Clinical Data Manager'), (u'New York, NY', u'NYU Langone Health System', None, u'Quality Improvement Project Manager'), (u'New York, NY 10005 (Financial District area)', u'WeWork', None, u'Senior Director/Director, Revenue Management Ops Research'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'NYU School of Medicine', None, u'Clinical Data Manager'), (u'New York, NY', u'NYU Langone Health System', None, u'Quality Improvement Project Manager'), (u'New York, NY 10005 (Financial District area)', u'WeWork', None, u'Senior Director/Director, Revenue Management Ops Research'), (u'New York, NY', u'Yahoo! Inc.', None, u'Research Engineer, Principal'), (u'New York, NY', u'NYU Langone Health System', None, u'Senior Radiology IT Analyst'), (u'New York, NY', u'Dailymotion', None, u'Senior Back End Engineer - Core Exchange'), (u'East Rutherford, NJ', u'JLL', None, u'Research Analyst'), (u'New York, NY', u'Microsoft', None, u'Data & Applied Scientist'), (u'New York, NY', u'Yahoo! Inc.', None, u'Research Engineer, Video'), (u'New York, NY', u'NYU School of Medicine', None, u'Research Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY', u'Bloomberg', None, u'Senior Machine Learning/NLP Software Engineer'), (u'New York, NY', u'BuzzFeed', None, u'Senior Research Manager, Ad Effectiveness (New York)'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Senior Scientist, Pharmacological Sciences'), (u'New York, NY', u'NYU School of Medicine', None, u'Financial Analyst'), (u'New York, NY', u'NYU Langone Health System', None, u'Senior Quality Improvement Project Coordinator'), (u'New York, NY', u'Crowded.com', None, u'Data Scientist/ Machine Learning Engineer at Fortu'), (u'Plainfield, NJ 07060', u'Katalyst Healthcares & Life Sciences', None, u'Data Manager'), (u'New York, NY', u'nuvento', None, u'Data Scientist'), (u'New York, NY 10013 (Tribeca area)', u'Galvanize', None, u'Instructor, Sr. Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Averity', None, u'Top NYC Hedge Fund ($20B+ AUM) looking for Senior Quantitati...'), (u'New York, NY', u'NYU School of Medicine', None, u'Financial Analyst'), (u'New York, NY', u'NYU Langone Health System', None, u'Senior Quality Improvement Project Coordinator'), (u'New York, NY', u'Crowded.com', None, u'Data Scientist/ Machine Learning Engineer at Fortu'), (u'Plainfield, NJ 07060', u'Katalyst Healthcares & Life Sciences', None, u'Data Manager'), (u'New York, NY', u'Averity', u'$150,000 - $300,000 a year', u'Quantitative Analyst for Global Quant Fund'), (u'New York, NY', u'nuvento', None, u'Data Scientist'), (u'New York, NY 10013 (Tribeca area)', u'Galvanize', None, u'Instructor, Sr. Data Scientist'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Senior Research Aide'), (u'New York, NY', u'TRC Companies Inc', None, u'Environmental Scientist - New York City, NY'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'Whippany, NJ', u'Oxford Solutions', None, u'Lead Statistical Analyst - Oncology'), (u'New York, NY 10013 (Tribeca area)', u'Galvanize', None, u'Instructor, Sr. Data Scientist'), (u'New York, NY', u'All-In Analytics', u'$140,000 - $150,000 a year', u'Senior Data Scientist'), (u'New York, NY', u'Amazon Corporate LLC', None, u'Software Development Manager \u2013 Emerging AWS Machine Learning'), (u'New York, NY', u'Elenion Technologies', None, u'Senior Optoelectronic Controls Engineer'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Artificial Intelligence Go to Market Senior Manager'), (u'New York, NY', u'Elti Solutions', None, u'Quantitative Analyst, global investments strategies'), (u'New York, NY', u'Spreemo', None, u'Senior Data Scientist'), (u'New York, NY 10003 (Greenwich Village area)', u'Medidata Solutions', None, u'Business Analytics Manager - Genomics'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR CLINICAL RESEARCH COORDINATOR - NEUROLOGY - FULL TIME...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'Carlstadt, NJ 07072', u'The Hartz Mountain Corporation', None, u'Lab Technician I'), (u'New York, NY', u'Deutsche Bank', None, u'Lead Software Engineer'), (u'New York, NY', u'Research Foundation of The City University of New...', None, u'Research Assistant'), (u'New York, NY', u'NYU School of Medicine', None, u'Clinical Data Manager'), (u'New York, NY', u'Sudler& Hennessey', None, u'Scientific Associate'), (u'Union Beach, NJ', u'IFF', None, u'Research Fellow, Analytical Research/Material Sciences'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Artificial Intelligence Go to Market Senior Manager'), (u'Jersey City, NJ', u'Verisk Insurance Solutions', None, u'Manager - Data Scientist'), (u'New York, NY 10019 (Midtown area)', u'Merkle Inc.', None, u'Spotfire Technical Lead & Trainer'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Recruiter, Quantitative Research'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'ASSOCIATE RESEARCHER I'), (u'New York, NY', u'LearnVest', None, u'Director of Marketing Analytics'), (u'New York, NY', u'Ambulatory/Outpatient NYU School of Medicine', None, u'FGP Sleep Technologist 1- Temporary'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Administrative Manager, Medical Physics'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'BIOINFORMATICIAN II - FULL TIME - DAYS - MSH'), (u'New York, NY', u'Dailymotion', None, u'Senior Back End Engineer - Core Exchange'), (u'Summit, NJ 07901', u'Celgene Corporation', None, u'Associate Scientist, Analytical Research & Development, Biol...'), (u'New York, NY', u'Hired by Matrix, Inc.', None, u'Quantitative Finance Analyst'), (u'New York, NY', u'Amazon Corporate LLC', None, u'Software Development Manager \u2013 Emerging AWS Machine Learning'), (u'New York, NY', u'Integrated Management Resources, LLC', None, u'Quantitative Analyst'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10010 (Gramercy area)', u'AppNexus', None, u'Senior Data Scientist'), (u'New York, NY 10005 (Financial District area)', u'Societe Generale Corporate and Investment Banking', None, u'Quantitative Analyst - Regulatory Model Design'), (u'Jersey City, NJ', u'Verisk Insurance Solutions', None, u'Data Scientist/Actuary'), (u'New York, NY', u'1010data', None, u'UX Developer'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'ASSOCIATE RESEARCHER I'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR RESIDENCY PROGRAM COORDINATOR'), (u'New York, NY', u'NYU School of Medicine', None, u'Division Administrator'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Manager, Tax Technology - Data Analytics'), (u'New York, NY', u'NYU Langone Health System', None, u'Epic Orders Analyst'), (u'New York, NY', u'Dailymotion', None, u'Senior Backend Engineer - Core Exchange'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'1010data', None, u'UX Developer'), (u'New York, NY 10018 (Clinton area)', u'Quartet', None, u'Clinical Community Specialist'), (u'New York, NY 10013 (Tribeca area)', u'Arena', None, u'Lead Software Engineer'), (u'New York, NY', u'NYU Langone Health System', None, u'Cisco Contact Center Engineer'), (u'New York, NY', u'Feedzai', None, u'Data Scientist'), (u'New York, NY', u'Columbia University', None, u'Sr Statistician SAS Programmer'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Associate Researcher II, Hematology Oncology Department, Can...'), (u'New York, NY 10005 (Financial District area)', u'Societe Generale Corporate and Investment Banking', None, u'Quantitative Analyst - Regulatory Model Design'), (u'New York, NY', u'Spreemo', None, u'Data Scientist, Statistician'), (u'Orangeburg, NY 10962', u'Instrumentation Laboratory', None, u'Summer Internship - Product Support'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10001 (Chelsea area)', u'Laguna Source', u'$100,000 a year', u'Senior Scientist, Upstream Mammalian Cell Culture Developmen...'), (u'New York, NY', u'Research Foundation of The City University of New...', None, u'Research Project Coordinator'), (u'New York, NY', u'Schrodinger', None, u'DevOps Engineer'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'ASSOCIATE RESEARCHER I - PULMONARY - FULL TIME - DAYS - MSH'), (u'New York, NY 10012 (Little Italy area)', u'Bond Street', None, u'Data Scientist'), (u'New York, NY', u'Ambulatory/Outpatient NYU School of Medicine', None, u'FGP Sec II-Intake/Sched Ob/Gyn IVF'), (u'New York, NY', u'Soci\xe9t\xe9 G\xe9n\xe9rale', None, u'Model Validation Quantitative Analyst - Equity'), (u'New York, NY', u'Amazon Corporate LLC', None, u'Software Development Engineer \u2013 Emerging AWS Machine Learnin...'), (u'New York, NY 10002 (Lower East Side area)', u'PricewaterhouseCoopers LLC', None, u'Data Scientist Manager - Advanced Risk & Compliance Analytic...'), (u'Berkeley Heights, NJ', u'Navitas', None, u'Statistician Medical Affair'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Privacy Compliance Manager'), (u'New York, NY', u'Hired by Matrix, Inc.', None, u'Quantitative Finance Analyst'), (u'New York, NY', u'Ambulatory/Outpatient NYU Hospitals Center', None, u'Manager-Operational Initiatives'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'CLINICAL PROGRAM MANAGER - SURGERY'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'BIOINFORMATICIAN II'), (u'Whippany, NJ', u'Recruiting Resources Co.', u'$61.25 an hour', u'Statistical Analyst (Oncology)'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Associate Researcher II, Hematology Oncology Department, Can...'), (u'New York, NY', u'Brooklyn Data Science', None, u'Data Scientist at FXcompared'), (u'New York, NY', u'Dynamics Associates', None, u'Senior Quantitative Analyst'), (u'New York, NY', u'Brooklyn Laboratory Charter School', None, u'Math Program Manager'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Loadsmart', None, u'Senior Data Scientist'), (u'Bronx, NY 10461 (Parkchester area)', u'Albert Einstein College of Medicine', None, u'Postdoctoral Fellow'), (u'Summit, NJ', u'Celgene', None, u'Sr. Clinical Research Scientist'), (u'New York, NY 10013 (Tribeca area)', u'T3 Trading Group LLC', None, u'Statistical Pairs Trading and Convergence Position Opening'), (u'New York, NY 10018 (Clinton area)', u'Schireson Associates', None, u'Software Engineer'), (u'Manhasset, NY', u'Northwell Health', None, u'Research Scientist (Bioinformatician) - Pediatric Genetics'), (u'Brooklyn, NY 11206 (Williamsburg area)', u'New York University', None, u'Project Manager, Center for K12 STEM Education'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'SENIOR CLINICAL RESEARCH COORDINATOR - NEUROLOGY - FULL TIME...'), (u'New York, NY', u'Brooklyn Data Science', None, u'Data Scientist at FXcompared'), (u'New York, NY 10004 (Financial District area)', u'W.R. Rosato & Associates', None, u'Sr. Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'2U', None, u'Data Scientist'), (u'New York, NY', u'Sudler& Hennessey', None, u'Scientific Associate'), (u'New York, NY', u'Impact Radius', None, u'Data Scientist (HC00467)'), (u'New York, NY 10020 (Midtown area)', u'Huxley Associates', None, u'Quantitative Risk Analyst'), (u'Manhattan, NY', u'DEPARTMENT OF FINANCE', u'$68,239 - $80,000 a year', u'Policy Data Analyst'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Data Control Assistant'), (u'New York, NY', u'OnDeck', None, u'Vice President, Product Management - Channel Experiences'), (u'New York, NY', u'Open Systems Technologies, Inc.', None, u'Trading Quantitative Analyst'), (u'New York, NY', u'NYU Langone Health System', None, u'Revenue Management Manager'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'McKinsey Academy Course Operations Coordinator'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Project Assistant IV'), (u'New York, NY', u'Impact Radius', None, u'Data Scientist (HC00467)'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Associate Researcher II, Metabolism Institute'), (u'New York, NY', u'2U', None, u'Data Scientist'), (u'Jersey City, NJ', u'Verisk Insurance Solutions', None, u'Principal Data Scientist'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Sr. Data Scientist'), (u'New York, NY', u'Facebook', None, u'Data Scientist, Analytics'), (u'New York, NY', u'Research Foundation of The City University of New...', None, u'Research Project Coordinator'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Data Control Assistant'), (u'New York, NY 10013 (Tribeca area)', u'New York Genome Center', None, u'Postdoctoral Researcher, Landau Lab'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'Jersey City, NJ', u'Verisk Insurance Solutions', None, u'Manager - Data Scientist'), (u'New York, NY 10011 (Chelsea area)', u'ReWork', None, u'Community Engagement Manager'), (u'New York, NY', u'Voya Financial', None, u'Quantitative Portfolio Analyst - Global Portfolio Solutions'), (u'New York, NY 10018 (Clinton area)', u'RapidSOS', None, u'Product Manager \u2013 Internet of Things'), (u'New York, NY', u'Capital One', None, u'Product Design Lead, Treasury Management'), (u'Plainfield, NJ 07060', u'Katalyst Healthcares & Life Sciences', None, u'Data Manager'), (u'Franklin Lakes, NJ 07417', u'Express Scripts', None, u'Sr. Hadoop Data Engineer'), (u'Brooklyn, NY 11206 (Williamsburg area)', u'New York University', None, u'Assistant Director, Center for K12 STEM Education'), (u'New York, NY 10032 (Washington Heights area)', u'Morgan Stanley', None, u'Senior Credit Risk Methodology Quantitative Analyst - Vice P...'), (u'Clark, NJ 07066', u"L'Oreal USA", None, u'Senior Research Scientist-Product Performance Evaluation'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'1010data', None, u'Director of Sales, Consumer Insights'), (u'New York, NY', u'WorldCover', u'$70,000 - $110,000 a year', u'Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Revenue Cycle Analyst'), (u'Kenilworth, NJ', u'Merck', None, u'Senior Scientist, Quantitative Pharmacology and Pharmacometr...'), (u'New York, NY', u'TEEMA', None, u'Director of Analytics - Leading FinTech Startup'), (u'New York, NY', u'Open Systems Technologies, Inc.', None, u'Machine Learning Developer'), (u'New York, NY 10002 (Lower East Side area)', u'PricewaterhouseCoopers LLC', None, u'Data Scientist Manager - Advanced Risk & Compliance Analytic...'), (u'Jersey City, NJ', u'InfoSmart Systems Inc', None, u'Data Scientist(USC and GC)'), (u'New York, NY', u'WSP | Parsons Brinckerhoff', None, u'Corporate Recruiter'), (u'New York, NY 10003 (Greenwich Village area)', u'Medidata Solutions', None, u'Business Analytics Manager - Genomics'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Dailymotion', None, u'Senior Backend Engineer - Core Exchange'), (u'New York, NY 10065 (Upper East Side area)', u'Memorial Sloan Kettering Cancer Center', None, u'Bioinformatics Data Scientist - Cancer Genomics'), (u'New York, NY 10010 (Gramercy area)', u'ERS Search', u'$175,000 a year', u'Lead Data Scientist w/ HR data'), (u'Mount Vernon, NY 10550', u'GERITREX LLC', None, u'Analytical Chemist'), (u'Hoboken, NJ', u'Jet', None, u'Data Scientist'), (u'New York, NY 10010 (Gramercy area)', u'Persado', None, u'Vice President, Financial Services'), (u'New York, NY', u'Schrodinger', None, u'DevOps Engineer'), (u'New York, NY', u'Tier1 IT', None, u'Backend Data Engineer'), (u'New York, NY', u'Columbia University', None, u'Director, Strategy and Partnerships'), (u'New York, NY', u'OnDeck', None, u'Vice President, Product Management - Channel Experiences'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'NYU School of Medicine', None, u'Clinicial Research Program Manager'), (u'New York, NY', u'Ambulatory/Outpatient NYU School of Medicine', None, u'FGP Sec II-Intake/Sched Ob/Gyn IVF'), (u'Parsippany, NJ', u'Tetra Tech', None, u'Assistant Scientist \u2013 Parsippany, NJ'), (u'New York, NY 10032 (Washington Heights area)', u'Morgan Stanley', None, u'Manager Solutions - Quantitative Portfolio Analyst'), (u'New York, NY 10013 (Tribeca area)', u'Arena', None, u'Senior Data Scientist'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Products Analytics Go To Market Manager'), (u'New York, NY', u'NBCUniversal', None, u'Data Scientist, Forecast Analyst'), (u'East Hanover, NJ 07936', u'Mondelez International', None, u'Quality Specialist \u2013 NA Biscuit'), (u'Hoboken, NJ', u'Genesis Research', None, u'Senior Analyst, Health Outcomes Research'), (u'New York, NY', u'Ambulatory/Outpatient NYU Hospitals Center', None, u'Manager-Operational Initiatives'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Custoria', None, u'Customer Success Lead'), (u'Jersey City, NJ', u'Mitsubishi Tanabe Pharma America', None, u'Clinical Research Scientist'), (u'New York, NY', u'TheNumber', None, u'Frontend Engineer'), (u'New York, NY 10003 (Greenwich Village area)', u'Integral Ad Science', None, u'Data Scientist'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Business Intelligence Developer II'), (u'Woodcliff Lake, NJ', u'BMW Financial Services, US', None, u'Senior Statistician'), (u'New York, NY', u'Capital One', None, u'Senior Quantitative Analyst \u2013 Model Implementation'), (u'Manhattan, NY', u'DEPT. OF HOMELESS SERVICES', u'$78,630 - $103,332 a year', u'Compliance Team Leader'), (u'New York, NY 10271 (Financial District area)', u'CAPCO', None, u'Data Scientist PC'), (u'New York, NY 10271 (Financial District area)', u'CAPCO', None, u'Data Scientist MP'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'NYU Langone Health System', None, u'Emergency Management Specialist'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Sr. Data Scientist'), (u'New York, NY', u'NYU Langone Health System', None, u'Director, Epic OpTime/Anesthesia'), (u'New York, NY', u'Cox Media Group', None, u'Financial Analyst - Gamut Smart Media (New York)'), (u'New York, NY', u'Schrodinger', None, u'Senior Scientist, Quantum Mechanics Applications'), (u'New York, NY', u'AbleTo, Inc.', None, u'Data Scientist'), (u'New York, NY', u'Custoria', None, u'Customer Success Lead'), (u'Summit, NJ', u'Celgene', None, u'Sr. Clinical Research Scientist'), (u'Brooklyn, NY 11249 (Chelsea area)', u'Vice Media Inc.', u'$11 an hour', u'Data Scientist Intern'), (u'New York, NY', u'Harnham', u'$150,000 a year', u'Data Scientist - Financial Services'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Go-to-Market, Financial Services - Risk Analytics Senior Man...'), (u'New York, NY', u'BuzzFeed', None, u'Client Services Manager, Pharma and Political, NYC'), (u'New York, NY', u'Isaacson Search Company', None, u'Data Scientist'), (u'New York, NY', u'BuzzFeed', None, u'Manager, Software Engineering- Tasty'), (u'New York, NY', u'NYU School of Medicine', None, u'Executive Assistant (Temp Program)'), (u'Manhasset, NY', u'Northwell Health', None, u'Research Scientist (Bioinformatician) - Pediatric Genetics'), (u'New York, NY', u'TheNumber', None, u'Frontend Engineer'), (u'New York, NY 10001 (Chelsea area)', u'The Nature Conservancy', None, u'Boat Launch Steward'), (u'New York, NY', u'Schrodinger', None, u'Software Engineer, Back End'), (u'New York, NY', u'NBCUniversal', None, u'Data Scientist, Forecast Analyst'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Isaacson Search Company', None, u'Data Scientist'), (u'New York, NY', u'BuzzFeed', None, u'Manager, Software Engineering- Tasty'), (u'New York, NY', u'NYU School of Medicine', None, u'Executive Assistant (Temp Program)'), (u'New York, NY 10017 (Midtown area)', u'Analytic Recruiting', None, u'Junior Risk Reporting Quantitative Analyst (SQL, VBA) Hedge...'), (u'Manhasset, NY', u'Northwell Health', None, u'Research Scientist (Bioinformatician) - Pediatric Genetics'), (u'New York, NY', u'TheNumber', None, u'Frontend Engineer'), (u'New York, NY 10001 (Chelsea area)', u'The Nature Conservancy', None, u'Boat Launch Steward'), (u'New York, NY', u'Schrodinger', None, u'Software Engineer, Back End'), (u'New York, NY', u'NBCUniversal', None, u'Data Scientist, Forecast Analyst'), (u'New York, NY', u'Eka Finance', None, u'Start Up Hiring Quant Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10018 (Clinton area)', u'RapidSOS', None, u'Product Manager \u2013 Internet of Things'), (u'New York, NY', u'BuzzFeed', None, u'Client Services Manager, Pharma and Political, NYC'), (u'New York, NY', u'Cox Media Group', None, u'Financial Analyst - Gamut Smart Media (New York)'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'POSTDOCTORAL FELLOW'), (u'New York, NY', u'TEEMA', None, u'Director of Analytics - Leading FinTech Startup'), (u'New York, NY', u'Open Systems Technologies, Inc.', None, u'Machine Learning Developer'), (u'New York, NY 10003 (Greenwich Village area)', u'Global Strategy Group', None, u'Data Scientist, Analytics'), (u'New York, NY', u'AiCure', None, u'Project Manager for AI Healthcare Company'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Analytics Go to Market, Products - Analyst'), (u'Madison, NJ', u'Merck', None, u'Data Manager Job'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Cox Media Group', None, u'Financial Analyst - Gamut Smart Media (New York)'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'POSTDOCTORAL FELLOW'), (u'New York, NY 10003 (Greenwich Village area)', u'Global Strategy Group', None, u'Data Scientist, Analytics'), (None, None, None, None), (u'New York, NY', u'AiCure', None, u'Project Manager for AI Healthcare Company'), (u'New York, NY 10011 (Chelsea area)', u'Accenture', None, u'Analytics Go to Market, Products - Analyst'), (u'Madison, NJ', u'Merck', None, u'Data Manager Job'), (u'Jersey City, NJ', u'EXL', None, u'Full Stack Software Developer'), (u'New York, NY', u'Soci\xe9t\xe9 G\xe9n\xe9rale', None, u'Quantitative Analyst - Quantitative Market Making'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Radiologist Assistant'), (u'New York, NY', u'DataCamp', None, u'Senior Data Science Course Developer (Python)'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'East Hanover, NJ 07936', u'Mondelez International', None, u'Research and Development Process Engineer'), (u'New York, NY', u'Earthstream Global, Inc', None, u'Senior Data Scientist'), (u'New York, NY 10001 (Chelsea area)', u'The Nature Conservancy', None, u'Boat Launch Steward'), (u'New York, NY', u'WSP | Parsons Brinckerhoff', None, u'Senior Environmental Geologist'), (u'Clark, NJ 07066', u"L'Oreal USA", None, u"L'Oreal, Director R&I Consumer & Market Insights (Hair)"), (u'New York, NY', u'Columbia University', u'$1,088 a week', u'Senior Research Worker - Endocrinology'), (u'New York, NY 10018 (Clinton area)', u'Quartet', None, u'Core Platform Engineer'), (u'New York, NY 10012 (Little Italy area)', u'Oscar Insurance', None, u'Senior Product Marketing Manager: Oscar for Business'), (u'Berkeley Heights, NJ', u'Aequor Technologies, Inc.', None, u'Statistician'), (u'Rahway, NJ', u'Merck', None, u'Senior Statistical Programmer Job'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'NYU Langone Health System', None, u'Revenue Management Manager'), (u'New York, NY', u'NBCUniversal', None, u'Senior Analyst, NBCU Entertainment Research & Strategy'), (u'New York, NY', u'Mitchell Martin Inc.', None, u'Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Financial Services Associate'), (u'Short Hills, NJ', u'Dun & Bradstreet', None, u'Statistician/Statistical Consultant - PL'), (u'New York, NY 10012 (Little Italy area)', u'Neustar, Inc.', None, u'Sr Principal Statistician'), (u'New York, NY', u'Schrodinger', None, u'Software Engineer, Back End'), (u'New York, NY', u'TheNumber', None, u'Business Analyst Intern'), (u'New York, NY', u'NYU School of Medicine', None, u'Associate General Counsel'), (u'New York, NY', u'Venturi Ltd', u'$120,000 - $170,000 a year', u'Big Data Engineer / Hadoop Developer ( FinTech / Hive / Hbas...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'NBCUniversal', None, u'Senior Analyst, NBCU Entertainment Research & Strategy'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Financial Services Associate'), (u'Short Hills, NJ', u'Dun & Bradstreet', None, u'Statistician/Statistical Consultant - PL'), (u'New York, NY 10012 (Little Italy area)', u'Neustar, Inc.', None, u'Sr Principal Statistician'), (u'New York, NY', u'Schrodinger', None, u'Software Engineer, Back End'), (u'New York, NY', u'TheNumber', None, u'Business Analyst Intern'), (u'New York, NY', u'NYU School of Medicine', None, u'Associate General Counsel'), (u'Totowa, NJ', u'Corbion', None, u'Senior Bakery Scientist, Frozen Dough'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Data Scientist - Advertising Product and Platforms'), (u'New York, NY 10022 (Midtown area)', u'McKinsey & Company', None, u'McKinsey Academy Course Operations Coordinator'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Harnham', u'$180,000 a year', u'Director, Data Scientist Modeler'), (u'New York, NY', u'Capital One', None, u'Product Design Lead, Treasury Management'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Cat Scan Technician/IV Cert - Temporary Fulltime'), (u'Union Beach, NJ', u'International Flavors and Fragrances', None, u'Research Fellow, Analytical Research/Material Sciences'), (u'New York, NY', u'Tier1 IT', None, u'Principal Engineer'), (u'Jamaica, NY 11434', u'Biocair, Inc', u'$20 - $23 an hour', u'Logistics Coordinator'), (u'New York, NY', u'Healthcare Consultancy Group', None, u'Account Manager - ProEd Communications'), (u'New York, NY', u'Soci\xe9t\xe9 G\xe9n\xe9rale', None, u'Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Senior NLP Software Engineer'), (u'New York, NY', u'1010data', None, u'Product Manager; Analytical Products'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'JPMorgan Chase', None, u'Legal- Firmwide Initiatives Team OLO Data Scientist- VP'), (u'New York, NY', u'PlaceIQ', None, u'Data Scientist'), (u'Woodcliff Lake, NJ', u'BMW Financial Services, US', None, u'Senior Statistician'), (u'New York, NY 10017 (Midtown area)', u'Analytic Recruiting', None, u'Junior Risk Reporting Quantitative Analyst (SQL, VBA) Hedge...'), (u'New York, NY 10010 (Gramercy area)', u'ERS Search', u'$175,000 a year', u'Lead Data Scientist w/ HR data'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Radiotherapy Technologist'), (u'Montvale, NJ', u'Pronix', u'$70 an hour', u'Data Scientist Hadoop'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Lab Courier'), (u'New York, NY', u'Columbia University', None, u'Post Doctoral Research Scientist'), (u'New York, NY', u'Capital One', None, u'Sr. Associate, Digital Product Management'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Senior Associate - Cognitive Data Scientist Natural Language...'), (u'New York, NY', u'Axius Technologies', None, u'Research Analyst / Senior Research Analyst - Trade Support'), (u'Florham Park, NJ', u'inVentiv Health Clinical', None, u'Senior Clinical Research Scientist'), (u'South Plainfield, NJ', u'Katalyst Healthcares & Life Sciences', None, u'Pharmacovigilance Scientist/ Researcher, Senior'), (u'New York, NY', u'NBCUniversal', None, u'Data Engineer'), (u'New York, NY', u'Elenion Technologies', None, u'Senior Laser Design Engineer'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Lab Courier'), (u'New York, NY', u'Ambulatory/Outpatient NYU School of Medicine', None, u'FGP Ultrasound Technician - Trinity *PER-DIEM* (GYN Specialt...'), (u'New York, NY', u'PlaceIQ', None, u'Data Scientist'), (u'New York, NY', u'Lorven Technologies', None, u'Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Elenion Technologies', None, u'Senior Laser Design Engineer'), (u'New York, NY', u'Averity', u'$125,000 - $150,000 a year', u'Data Scientist for Multi-Million Dollar, Data- Driven Health...'), (u'New York, NY 10011 (Chelsea area)', u'Natural Resources Defense Council', None, u'SALESFORCE ADMINISTRATOR'), (u'New York, NY 10017 (Midtown area)', u'Wunderman', None, u'Senior Project Manager-RX'), (u'New York, NY', u'OkCupid', None, u'Product Designer'), (u'New York, NY', u'Averity', u'$110,000 - $130,000 a year', u'Data Scientist (Media Company)'), (u'Port Washington, NY', u'NPD', None, u'Data Quality/Scientist/Analyst role'), (u'New York, NY', u'Viacom', None, u'SENIOR DATA SCIENTIST'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Account Representative - LaGuardia Medical Billing'), (u'New York, NY', u'NYU Langone Health System', None, u'Associate Director, Decision Support'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'DIRECTOR'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Budget & Reporting Analyst II'), (u'New York, NY', u'NYU School of Medicine', None, u'Director, IT Research Administration'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'CLINICAL TRIALS MANAGER'), (u'Berkeley Heights, NJ', u'Katalyst Healthcares & Life Sciences', None, u'Statistical Programmer'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Data Scientist, Natural Language Processing'), (None, None, None, None), (u'New York, NY', u'Viacom', None, u'SENIOR DATA SCIENTIST'), (u'New York, NY', u'Columbia University', u'$886 a week', u'Research Assistant - Nephrology'), (u'Whippany, NJ', u'CTG', None, u'Lead Statistical Analyst - Oncology - 17202502'), (u'New York, NY 10007 (Financial District area)', u'The New York Academy of Sciences', None, u'Program Coordinator, Takeda Innovators in Science Award'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10001 (Chelsea area)', u'Comcast', None, u'Eng 2, Software Dev & Engineering'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Data Scientist, Natural Language Processing'), (u'New York, NY', u'NYU School of Medicine', None, u'FGP Ultrasound Technician (part time), Greenpoint, Brooklyn'), (u'New York, NY 10012 (Little Italy area)', u'UDig', u'$50 - $70 an hour', u'Senior Business/Data Analyst'), (u'New York, NY 10018 (Clinton area)', u'JW Player', None, u'Principal Software Engineer'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Sr. Software Engineer - Healthcare & Life Sciences'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Grants Administrator'), (u'New York, NY', u'Blackwood Seven', None, u'Data Scientist'), (u'New York, NY', u'NYU Langone Health System', None, u'Senior Clinical Engineer'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Clinical Lab Technician (Immunology)'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'Whippany, NJ', u'Belcan TechServices', None, u'Lead Statistical Analyst job in Whippany, NJ!'), (u'Wayne, NJ', u'Valley National Bank', None, u'Quantitative Business Analyst'), (u'New York, NY', u'Trov', None, u'Data Scientist'), (u'New York, NY 10012 (Little Italy area)', u'24 Seven', None, u'Statistician'), (u'New York, NY 10005 (Financial District area)', u'Hudson River Trading', None, u'Desktop Administrator Intern'), (u'Union Beach, NJ', u'IFF', None, u'Senior Research Investigator, Chemical Information and Model...'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Sr. Data Scientist - Customer Segmentation'), (u'Murray Hill, NJ', u'C.R. Bard International', None, u'Clinical Data Scientist'), (u'New York, NY 10016 (Gramercy area)', u'NYU Hospitals Center', None, u'Pharmacy Intern'), (u'New York, NY', u'Averity', u'$140,000 - $150,000 a year', u'Data Scientist for Computer Vision Company'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10012 (Little Italy area)', u'24 Seven', None, u'Director of Analytics'), (u'New York, NY 10285 (Battery Park area)', u'Reperio Inc.', None, u'Data Scientists & Predictive Modeling Consultants'), (u'Summit, NJ 07901', u'Celgene Corporation', None, u'Sr. Clinical Research Scientist - Multiple Myeloma'), (u'Kenilworth, NJ 07033', u'TalentBurst, Inc.', None, u'Scientist - III (Senior)'), (u'New York, NY', u'Weill Cornell Medicine Vein Treatment Center', None, u'Registered Vascular Technologist'), (u'New York, NY', u'Axius Technologies', None, u'Research Analyst / Senior Research Analyst - Trade Support'), (None, None, None, None), (u'New York, NY', u'AmTrust Financial Services', None, u'Quantitative Business Analyst'), (u'New York, NY', u'OnDeck', None, u'Senior Product Manager'), (u'Summit, NJ', u'Celgene', None, u'Senior Scientist, Translational Medicine'), (u'New York, NY', u'BuzzFeed', None, u'HRIS Analyst'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'BuzzFeed', None, u'Breaking News Reporter'), (u'New York, NY', u'TheNumber', None, u'Business Analyst Intern'), (u'Manhattan, NY', u'DEPT OF HEALTH/MENTAL HYGIENE', u'$87,768 - $109,008 a year', u'Unit Chief/Bureau of Public Health Laboratory'), (u'New York, NY', u'BuzzFeed for Video Internship/Fellowship/Residency', None, u'Video Fellow (NYC)'), (u'New York, NY', u'Columbia University', None, u'System Administrator'), (u'New York, NY', u'Feedzai', None, u'Sr Software Engineer'), (u'New York, NY', u'Schrodinger', None, u'Software Engineer, Front End'), (u'New York, NY', u'CompIQ', None, u'Data Scientist'), (u'New York, NY 10271 (Financial District area)', u'Brennan Center for Justice', None, u'Research and Program Associate, Justice Program'), (u'New York, NY', u'Averity', u'$125,000 - $150,000 a year', u'Data Scientist for Multi-Million Dollar, Data- Driven Health...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Averity', u'$125,000 - $150,000 a year', u'Data Scientist for Multi-Million Dollar, Data- Driven Health...'), (u'New York, NY 10017 (Midtown area)', u'Wunderman', None, u'Senior Project Manager'), (u'New York, NY', u'Springer Nature', None, u'Assistant Editor, Operations Research and Management Science'), (u'New York, NY', u'Capital One', None, u'Sr. Associate, Digital Product Management'), (u'New York, NY', u'Averity', u'$110,000 - $130,000 a year', u'Data Scientist (Media Company)'), (u'New York, NY', u'Two Sigma Investments, LLC.', None, u'Legal & Compliance Strategy Analyst'), (u'Florham Park, NJ', u'Conduent', None, u'Senior Quantitative Financial Analyst'), (u'New York, NY 10017 (Midtown area)', u'Wunderman', None, u'Senior Project Manager-RX'), (u'Jersey City, NJ', u'Forbes Media LLC', None, u'Writer/Producer, Editorial Special Features'), (u'East Hanover, NJ 07936', u'Aequor Technologies', None, u'Global Trial Leader'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'BuzzFeed for Video Internship/Fellowship/Residency', None, u'Video Fellow (NYC)'), (u'New York, NY', u'Columbia University', None, u'System Administrator'), (u'New York, NY', u'Feedzai', None, u'Sr Software Engineer'), (u'New York, NY', u'Schrodinger', None, u'Software Engineer, Front End'), (u'New York, NY', u'CompIQ', None, u'Data Scientist'), (u'New York, NY 10271 (Financial District area)', u'Brennan Center for Justice', None, u'Research and Program Associate, Justice Program'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Sr. Data Scientist - Customer Segmentation'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Lab Information Management System Specialist'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Senior Patient Care Coordinator'), (u'New York, NY', u'Weill Cornell Medical College', None, u'Departmental Assistant'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'Summit, NJ', u'Aequor Technologies', None, u'Scientist, Clinical Research'), (u'New York, NY 10003 (Greenwich Village area)', u'Nielsen', None, u'Data Scientist - Nielsen Marketing Cloud'), (u'Brooklyn, NY 11201', u'Renaissance Learning, Inc.', None, u'Technical Lead'), (u'Berkeley Heights, NJ', u'Celgene', None, u'Senior Statistician'), (u'Berkeley Heights, NJ', u'Aequor Technologies', None, u'SAS Statistical Programmer'), (u'New York, NY', u'Arena.io', None, u'Software Engineer'), (u'New York, NY', u'Columbia University', None, u'Post Doctoral Research Scientist'), (u'New York, NY 10154 (Midtown area)', u'KPMG', None, u'Sr. Software Engineer - Healthcare & Life Sciences'), (u'New York, NY', u'DemystData', u'$75,000 - $110,000 a year', u'Manager, Client Strategy'), (u'New York, NY', u'NYU Langone Health System', None, u'Environmental Health and Safety Specialist II'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10011 (Chelsea area)', u'Spotify', None, u'Data Scientist (Product)'), (u'New York, NY 10279 (Financial District area)', u'FactSet Research Systems', None, u'Data Scientist'), (u'East Hanover, NJ 07936', u'Mondelez International', None, u'Sr. Scientist I Specifications'), (u'New York, NY', u'Open Systems Technologies, Inc.', None, u'Research Business Analyst'), (u'Summit, NJ', u'Aequor Technologies', None, u'Scientist, Clinical Research'), (u'New York, NY', u'Harnham', u'$140,000 a year', u'Senior Big Data Engineer'), (u'New York, NY 10003 (Greenwich Village area)', u'Nielsen', None, u'Data Scientist - Nielsen Marketing Cloud'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Manager of Advanced Analytics'), (u'Brooklyn, NY 11201', u'Renaissance Learning, Inc.', None, u'Technical Lead'), (u'Kenilworth, NJ', u'Zillion Technologies', None, u'Senior Scientist (cell biology/molecular biology)'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY', u'Ezra Penland Actuarial Recruitment', None, u'Actuary with Predictive Modeling and Programming Skills #721...'), (u'New York, NY', u'Execu | Search', u'$150,000 - $200,000 a year', u'Quantitative Analyst'), (u'New York, NY', u'Harnham', u'$180,000 a year', u'Director, Data Scientist Modeler'), (u'New York, NY', u'Columbia University', None, u'Statistical Programmer/Data Manager - General Medicine'), (u'New York, NY', u'Harnham', u'$220,000 a year', u'Chief Data Scientist - Consulting'), (u'New York, NY', u'BuzzFeed', None, u'Site Reliability Engineer'), (u'Whippany, NJ', u'Belcan TechServices', None, u'Lead Statistical Analyst job in Whippany, NJ!'), (u'Newark, NJ', u'R&D Partners', None, u'Research Scientist'), (u'Manhasset, NY 11030', u'Northwell Health', None, u'Neural Computer Engineer - Bioelectronic Medicine'), (u'New York, NY', u'AmTrust Financial Services', None, u'Quantitative Business Analyst'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'Rahway, NJ', u'Merck', None, u'Document Specialist, Regulatory Affairs CMC Job'), (u'New York, NY', u'Delve', None, u'Project Manager and Research Analyst Intern'), (u'East Hanover, NJ 07936', u'Whiz Finder Corporation', None, u'Manager Statistical Programming'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Data Scientist (Contract)'), (u'New York, NY', u'Ezra Penland Actuarial Recruitment', None, u'Actuary with Predictive Modeling and Programming Skills #721...'), (u'New York, NY', u'Xaxis', None, u'Product Manager - CoPilot'), (u'New York, NY', u'Harnham', u'$220,000 a year', u'Chief Data Scientist - Consulting'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Data Scientist (Contract to Hire)'), (u'New York, NY', u'Tapad', None, u'Data Scientist'), (u'Florham Park, NJ 07932', u'PricewaterhouseCoopers LLC', None, u'Learning & Development Measurement & Analytics Data Scientis...'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY 10011 (Chelsea area)', u'ReWork', None, u'Finance & Operations Manager'), (u'Harlem, NY', u'Harlem Village Academies', None, u'High School Science Teacher 2017-18'), (u'New York, NY', u'Tapad', None, u'Data Scientist'), (u'New York, NY', u'Sudler& Hennessey', None, u'Copywriter'), (u'New York, NY', u'Collective', None, u'Product Manager'), (u'New York, NY', u'JBCConnect', None, u'Senior Quantitative Analyst'), (u'New York, NY', u'Axius Technologies', None, u'Research Analyst / Senior Research Analyst - Trade Support'), (u'New York, NY', u'NYU School of Medicine', None, u'Finance Project Manager-Business Intelligence (Research Miss...'), (u'Summit, NJ 07901', u'Career Developers', None, u'Senior Safety Scientist Consultant'), (u'New York, NY', u'Countr', None, u'Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Columbia University', u'$886 a week', u'Senior Technician - CLM'), (u'New York, NY 10119 (Chelsea area)', u'IQ Workforce', None, u'Data Scientist (Contract)'), (u'New York, NY', u'Dia&Co', None, u'Data Scientist'), (u'New York, NY 10001 (Chelsea area)', u'Uber', None, u'Product Designer, Observability'), (u'New York, NY 10003 (Greenwich Village area)', u'Knewton', None, u'Data Scientist'), (u'New York, NY 10012 (Little Italy area)', u'24 Seven', None, u'Director of Analytics'), (u'New York, NY', u'Climb', None, u'Marketing Manager'), (u'New York, NY 10029 (Yorkville area)', u'Mount Sinai Health System', None, u'Biomedical Software Developer - Neurology - Full Time - Days...'), (u'Astoria, NY', u'NYU School of Medicine', None, u'FGP Patient Care Assistant (35), Astoria'), (u'New York, NY', u'Oracle', None, u'Product Mgmt/Strategy Snr Director-ProdDev'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist'), (u'New York, NY 10003', u'Verizon', None, u'Senior Data Scientist'), (u'New York, NY 10017', u'Aetna', None, u'Principal Data Scientist'), (u'New York, NY', u'Bloomberg', None, u'Applied Machine Learning Scientist/Engineer - Relevance & Di...'), (u'New York, NY', u'Bloomberg', None, u'Software Engineer / Research Scientist - Machine Learning Te...'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist - Big Data & Analytics'), (u'New York, NY 10001', u'BlackRock', None, u'People Analytics & Research Data Scientist'), (u'New York, NY 10154', u'KPMG', None, u'Data Scientist'), (u'New York, NY', u'Memorial Sloan Kettering', None, u'Sr. Strategic Analyst ( Data Scientist )'), (u'New York, NY', u'TD Bank', None, u'Quantitative Analyst III - Model Risk Execution'), (u'United States', u'Predictive Science', None, u'Data Scientist'), (u'New York, NY', u"L'Oreal USA", None, u"L'Oreal USA, Data Activation & DMP Lead"), (u'New York, NY', u'State Street', None, u'Quantitative Analyst, Global Analyst, Global Exchange, Offic...'), (u'New York, NY', u'OppenheimerFunds', None, u'AVP Intermediate Research Analyst'), (u'New York, NY', u'Bloomberg', None, u'Data Scientist - Network Infrastructure'), (u'New York, NY', u'WorkFusion', None, u'Deep Learning Data Scientist')]


## Create Dataframe DF 4  features
** 1- Location **

** 2- Company **

** 3- Salary **

** 4- Title **


```python
df= pd.DataFrame(results,columns=['location','company','salary','title'])
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>company</th>
      <th>salary</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New York, NY 10154</td>
      <td>KPMG</td>
      <td>None</td>
      <td>Data Scientist - Big Data &amp; Analytics</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New York, NY 10017</td>
      <td>Aetna</td>
      <td>None</td>
      <td>Principal Data Scientist</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New York, NY</td>
      <td>Bloomberg</td>
      <td>None</td>
      <td>Software Engineer / Research Scientist - Machi...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York, NY 10022 (Midtown area)</td>
      <td>McKinsey &amp; Company</td>
      <td>None</td>
      <td>Associate - Machine Learning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>New York, NY</td>
      <td>Twitter</td>
      <td>None</td>
      <td>Machine Learning</td>
    </tr>
  </tbody>
</table>
</div>



## CLEAN DATA SET
* Location Fixer
* Salary Fixer
* Add City
* Add State


```python
#APPLY LOCATION FIXER
df["location"]= df['location'].map(lambda x: None if x==None else location_fixer(x))
```


```python
# ADD CITY COLUMN
df['city']= df['location'].map(lambda x: x.split(",")[0] if x else None)
```


```python
# ADD STATE COLUMN
df['state']= df['location'].map(lambda x: x.split(",")[-1] if x else None)
```


```python
# APPLY SALARY FIX
df["salary"]= df['salary'].map(lambda x: None if x==None else fix_salaries(x))
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3006 entries, 0 to 3005
    Data columns (total 6 columns):
    location    3000 non-null object
    company     2998 non-null object
    salary      96 non-null float64
    title       3000 non-null object
    city        3000 non-null object
    state       3000 non-null object
    dtypes: float64(1), object(5)
    memory usage: 141.0+ KB


## Shockingly only 116 out of 3004 listings contain salary information!!
* Less than 5% of listings contain salary information, which creates obvious limitations for efficacy of the model
* Additiionally worth noting here that with more time would have like to add additional cities/regions and in\n
in doing so would introduce a meaningful geographic element to the data and analysis
* Lastly would have added additinal Count Vectorizer variables like job description
* Additional researhc on further variables to include beyond keywords
* Place emphasis on certain keywords
* Industry classification
* Classify by job requirements (GPA, degree, work experience)


```python
#count of salaries that are greater than zero-- using this set for training/testing
print "Only %.i/3004 listings have salary detail!"%len(df[df['salary'].isnull()==False])
```

    Only 96/3004 listings have salary detail!


## CREATE NEW DATA FRAME FOR TEST, 'TESTDATA'

### Only using dataset where salary info is provided


```python
#CREATED NEW DATA FRAME FOR TEST, USING SUBSET WITH SALARIES ONLY
testdata= df[~df['salary'].isnull()]
```

-- Dropping any entries where Title is null/empty to avoid issues with the Count Vectorizer 


```python
#drop na values in title column
testdata.dropna(subset=['title'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>company</th>
      <th>salary</th>
      <th>title</th>
      <th>city</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>New York, NY</td>
      <td>Venturi Ltd</td>
      <td>200000.0</td>
      <td>Data Scientist ( FinTech / Python / R / Machin...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>34</th>
      <td>New York, NY</td>
      <td>Research Foundation of The City University of ...</td>
      <td>36476.0</td>
      <td>Research associate</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>108</th>
      <td>New York, NY</td>
      <td>DEPT OF HEALTH/MENTAL HYGIENE</td>
      <td>59708.0</td>
      <td>Data Analyst, Bureau of Immunization</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>111</th>
      <td>New York, NY</td>
      <td>Research Foundation of The City University of ...</td>
      <td>80000.0</td>
      <td>Senior Research Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>113</th>
      <td>New York, NY</td>
      <td>POLICE DEPARTMENT</td>
      <td>42288.0</td>
      <td>Statistician, Level I</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>114</th>
      <td>New York, NY</td>
      <td>Access Staffing LLC</td>
      <td>180000.0</td>
      <td>Quantitative Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Manhattan, NY</td>
      <td>DEPARTMENT OF FINANCE</td>
      <td>70286.0</td>
      <td>Data Analyst/Modeler</td>
      <td>Manhattan</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>147</th>
      <td>New York, NY</td>
      <td>Urban Scholars Program, City College of New York</td>
      <td>80000.0</td>
      <td>Computer Science (Data Analysis) Instructor</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>153</th>
      <td>Manhattan, NY</td>
      <td>ADMIN FOR CHILDREN'S SVCS</td>
      <td>70286.0</td>
      <td>Data Analyst</td>
      <td>Manhattan</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>157</th>
      <td>New York, NY</td>
      <td>Urban Scholars Program, City College of New York</td>
      <td>80000.0</td>
      <td>Computer Science (Data Analysis) Instructor</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>158</th>
      <td>New York, NY</td>
      <td>Analytic Recruiting</td>
      <td>160000.0</td>
      <td>Senior Data Scientist</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>171</th>
      <td>Manhattan, NY</td>
      <td>ADMIN FOR CHILDREN'S SVCS</td>
      <td>70286.0</td>
      <td>Data Analyst</td>
      <td>Manhattan</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>186</th>
      <td>New York, NY</td>
      <td>Analytic Recruiting</td>
      <td>160000.0</td>
      <td>Senior Data Scientist</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>191</th>
      <td>New York, NY</td>
      <td>Enterprise Select</td>
      <td>180000.0</td>
      <td>Sr Quantitative Finance Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>202</th>
      <td>New York, NY</td>
      <td>DEPT OF HEALTH/MENTAL HYGIENE</td>
      <td>70286.0</td>
      <td>Data Manager – Researcher, Bureau of Children,...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>230</th>
      <td>New York, NY</td>
      <td>Enterprise Select</td>
      <td>180000.0</td>
      <td>Sr Quantitative Finance Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>231</th>
      <td>New York, NY</td>
      <td>Kennedy Unlimited Inc, Professional Staffing</td>
      <td>130000.0</td>
      <td>Predictive Analytics (Machine Learning)</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>232</th>
      <td>New York, NY</td>
      <td>Darwin Recruitment</td>
      <td>240000.0</td>
      <td>Machine Learning Engineer - NLP - Java - Pytho...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>236</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>180000.0</td>
      <td>Senior Data Scientist - NLP and Machine Learning</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>247</th>
      <td>New York, NY</td>
      <td>Enterprise Select</td>
      <td>140000.0</td>
      <td>Senior Data Engineer</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>262</th>
      <td>New York, NY</td>
      <td>Enterprise Select</td>
      <td>130000.0</td>
      <td>Data Scientist</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>266</th>
      <td>New York, NY</td>
      <td>Kennedy Unlimited Inc, Professional Staffing</td>
      <td>130000.0</td>
      <td>Predictive Analytics (Machine Learning)</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>267</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>180000.0</td>
      <td>Senior Data Scientist - NLP and Machine Learning</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>279</th>
      <td>Berkeley Heights, NJ</td>
      <td>Aequor Technologies</td>
      <td>140000.0</td>
      <td>Statistical Programmer/ Statistician Medical A...</td>
      <td>Berkeley Heights</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>280</th>
      <td>New York, NY</td>
      <td>Kennedy Unlimited Inc, Professional Staffing</td>
      <td>130000.0</td>
      <td>Predictive Analytics (Machine Learning)</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>289</th>
      <td>New York, NY</td>
      <td>Darwin Recruitment</td>
      <td>240000.0</td>
      <td>Machine Learning Engineer - NLP - Java - Pytho...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>290</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>180000.0</td>
      <td>Senior Data Scientist - NLP and Machine Learning</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>304</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>150000.0</td>
      <td>Senior Data Scientist - Modeling</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>319</th>
      <td>New York, NY</td>
      <td>Enterprise Select</td>
      <td>140000.0</td>
      <td>Senior Data Engineer</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>325</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>150000.0</td>
      <td>Data Engineer</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>980</th>
      <td>New York, NY</td>
      <td>Laguna Source</td>
      <td>100000.0</td>
      <td>Senior Scientist, Upstream Mammalian Cell Cult...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>Whippany, NJ</td>
      <td>Recruiting Resources Co.</td>
      <td>122500.0</td>
      <td>Statistical Analyst (Oncology)</td>
      <td>Whippany</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>Manhattan, NY</td>
      <td>DEPARTMENT OF FINANCE</td>
      <td>68239.0</td>
      <td>Policy Data Analyst</td>
      <td>Manhattan</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>New York, NY</td>
      <td>WorldCover</td>
      <td>70000.0</td>
      <td>Data Scientist</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>New York, NY</td>
      <td>ERS Search</td>
      <td>175000.0</td>
      <td>Lead Data Scientist w/ HR data</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1122</th>
      <td>Manhattan, NY</td>
      <td>DEPT. OF HOMELESS SERVICES</td>
      <td>78630.0</td>
      <td>Compliance Team Leader</td>
      <td>Manhattan</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>Brooklyn, NY</td>
      <td>Vice Media Inc.</td>
      <td>22000.0</td>
      <td>Data Scientist Intern</td>
      <td>Brooklyn</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>150000.0</td>
      <td>Data Scientist - Financial Services</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>New York, NY</td>
      <td>Columbia University</td>
      <td>1088.0</td>
      <td>Senior Research Worker - Endocrinology</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1230</th>
      <td>New York, NY</td>
      <td>Venturi Ltd</td>
      <td>120000.0</td>
      <td>Big Data Engineer / Hadoop Developer ( FinTech...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1251</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>180000.0</td>
      <td>Director, Data Scientist Modeler</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1256</th>
      <td>Jamaica, NY</td>
      <td>Biocair, Inc</td>
      <td>40000.0</td>
      <td>Logistics Coordinator</td>
      <td>Jamaica</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>New York, NY</td>
      <td>ERS Search</td>
      <td>175000.0</td>
      <td>Lead Data Scientist w/ HR data</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1272</th>
      <td>Montvale, NJ</td>
      <td>Pronix</td>
      <td>140000.0</td>
      <td>Data Scientist Hadoop</td>
      <td>Montvale</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>1297</th>
      <td>New York, NY</td>
      <td>Averity</td>
      <td>125000.0</td>
      <td>Data Scientist for Multi-Million Dollar, Data-...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>New York, NY</td>
      <td>Averity</td>
      <td>110000.0</td>
      <td>Data Scientist (Media Company)</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1319</th>
      <td>New York, NY</td>
      <td>Columbia University</td>
      <td>886.0</td>
      <td>Research Assistant - Nephrology</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1330</th>
      <td>New York, NY</td>
      <td>UDig</td>
      <td>100000.0</td>
      <td>Senior Business/Data Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1351</th>
      <td>New York, NY</td>
      <td>Averity</td>
      <td>140000.0</td>
      <td>Data Scientist for Computer Vision Company</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1375</th>
      <td>Manhattan, NY</td>
      <td>DEPT OF HEALTH/MENTAL HYGIENE</td>
      <td>87768.0</td>
      <td>Unit Chief/Bureau of Public Health Laboratory</td>
      <td>Manhattan</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1382</th>
      <td>New York, NY</td>
      <td>Averity</td>
      <td>125000.0</td>
      <td>Data Scientist for Multi-Million Dollar, Data-...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1388</th>
      <td>New York, NY</td>
      <td>Averity</td>
      <td>125000.0</td>
      <td>Data Scientist for Multi-Million Dollar, Data-...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1392</th>
      <td>New York, NY</td>
      <td>Averity</td>
      <td>110000.0</td>
      <td>Data Scientist (Media Company)</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1426</th>
      <td>New York, NY</td>
      <td>DemystData</td>
      <td>75000.0</td>
      <td>Manager, Client Strategy</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1438</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>140000.0</td>
      <td>Senior Big Data Engineer</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>New York, NY</td>
      <td>Execu | Search</td>
      <td>150000.0</td>
      <td>Quantitative Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>180000.0</td>
      <td>Director, Data Scientist Modeler</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>220000.0</td>
      <td>Chief Data Scientist - Consulting</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>New York, NY</td>
      <td>Harnham</td>
      <td>220000.0</td>
      <td>Chief Data Scientist - Consulting</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1493</th>
      <td>New York, NY</td>
      <td>Columbia University</td>
      <td>886.0</td>
      <td>Senior Technician - CLM</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 6 columns</p>
</div>




```python
#RESET INDEX IN TEST DATA
# TO AVOID ISSUES WHEN CONCATENATING THE DATA
testdata.reset_index(inplace=True)
```


```python
testdata.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>location</th>
      <th>company</th>
      <th>salary</th>
      <th>title</th>
      <th>city</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>New York, NY</td>
      <td>Venturi Ltd</td>
      <td>200000.0</td>
      <td>Data Scientist ( FinTech / Python / R / Machin...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>New York, NY</td>
      <td>Research Foundation of The City University of ...</td>
      <td>36476.0</td>
      <td>Research associate</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>108</td>
      <td>New York, NY</td>
      <td>DEPT OF HEALTH/MENTAL HYGIENE</td>
      <td>59708.0</td>
      <td>Data Analyst, Bureau of Immunization</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>111</td>
      <td>New York, NY</td>
      <td>Research Foundation of The City University of ...</td>
      <td>80000.0</td>
      <td>Senior Research Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113</td>
      <td>New York, NY</td>
      <td>POLICE DEPARTMENT</td>
      <td>42288.0</td>
      <td>Statistician, Level I</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
  </tbody>
</table>
</div>




```python
testdata['location'].nunique()
```




    8



#### Have 8 unique values for locations, not too bad to manage


```python
testdata['location'].value_counts()
```




    New York, NY            72
    Manhattan, NY           17
    East Hanover, NJ         2
    Berkeley Heights, NJ     1
    Whippany, NJ             1
    Montvale, NJ             1
    Jamaica, NY              1
    Brooklyn, NY             1
    Name: location, dtype: int64




```python
testdata.title.shape
```




    (96,)




```python
testdata.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>location</th>
      <th>company</th>
      <th>salary</th>
      <th>title</th>
      <th>city</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>New York, NY</td>
      <td>Venturi Ltd</td>
      <td>200000.0</td>
      <td>Data Scientist ( FinTech / Python / R / Machin...</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>New York, NY</td>
      <td>Research Foundation of The City University of ...</td>
      <td>36476.0</td>
      <td>Research associate</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>108</td>
      <td>New York, NY</td>
      <td>DEPT OF HEALTH/MENTAL HYGIENE</td>
      <td>59708.0</td>
      <td>Data Analyst, Bureau of Immunization</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>111</td>
      <td>New York, NY</td>
      <td>Research Foundation of The City University of ...</td>
      <td>80000.0</td>
      <td>Senior Research Analyst</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113</td>
      <td>New York, NY</td>
      <td>POLICE DEPARTMENT</td>
      <td>42288.0</td>
      <td>Statistician, Level I</td>
      <td>New York</td>
      <td>NY</td>
    </tr>
  </tbody>
</table>
</div>



## CREATE TRAIN/TEST SETS


```python
##SPLIT DATA INTO TRAIN/TEST SETS
# 67% train, 33% test

colvars=[x for x in testdata.columns if x not in ['salary', 'index', 'city']]

X= testdata[colvars]
y=testdata['salary']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.33)
X_train.reset_index(inplace=True)
X_test.reset_index(inplace=True)
```

## COUNT VECTORIZER ON TITLE IN X_TRAIN DATASET
* Create new features/variables for top 50 words that appear in 'Title' string for the Training Dataset
* Perform fit_transform
* Create new dataframe Xtrain


```python
# Count vectorizer on top 50 most frequent words in Title for XTRAIN DATASET
v = CountVectorizer(
    binary=True,  # Create binary features
    lowercase=True,
    stop_words='english', # Ignore common words such as 'the', 'and'
    max_features=50, # Only use the top 50 most common words
)

#fit transform to training set, save as X
Xtrain = v.fit_transform(X_train.title.values).todense() 
##ONLY X_train
Xtrain = pd.DataFrame(Xtrain, columns=v.get_feature_names())
```


```python
Xtrain.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>affairs</th>
      <th>analysis</th>
      <th>analyst</th>
      <th>assistant</th>
      <th>big</th>
      <th>bureau</th>
      <th>center</th>
      <th>chief</th>
      <th>company</th>
      <th>computer</th>
      <th>...</th>
      <th>science</th>
      <th>scientist</th>
      <th>senior</th>
      <th>sr</th>
      <th>statistical</th>
      <th>statistician</th>
      <th>team</th>
      <th>technician</th>
      <th>trade</th>
      <th>world</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



###  Dummify Training Set Features
* Dummify 'location' and 'company' features
* Drop_first columm 
* Concatenate Xtrain and dummies into Xtrain


```python
#dummify location + company (be sure to remove one variable/column)
#concatenate into the testdata df
categories=['location', 'company', 'state']
for cat in categories:
    series = X_train[cat]
    dummies= pd.get_dummies(series, prefix=cat, drop_first=True)
    Xtrain=pd.concat([Xtrain,dummies], axis=1) #just train set
```


```python
print "Xtrain shape: "
Xtrain.shape
```

    Xtrain shape: 





    (64, 91)



### Transform X_test data
* Transform only, NOT fit_transform as 'v' already knows which features have dummies
* Create Xtest dataframe


```python
Xtest = v.transform(X_test.title.values).todense() 
Xtest = pd.DataFrame(Xtest, columns=v.get_feature_names())  
```

### Dummify Test Data Features
* 'Location', 'Company', and 'State' features to be dummified
*  Need to clean columns
    - MISSING Columns to X_train need to be added and zero out values
    - EXTRA columsn to X_train need to be deleted
* Concatenate Xtest and dummies to Xtest
* Will have to ensure this Xtest DF has the EXACT same columns as the Xtrain DF
    - And in the same order! 


```python
categories=['location', 'company', 'state']
for cat in categories:
    series = X_test[cat]
    dummies= pd.get_dummies(series, prefix=cat, drop_first=True)
    Xtest=pd.concat([Xtest,dummies], axis=1)
```

* compare Xtest columns to Xtrain columns


```python
Xtest_columns= Xtest.columns
print " Xtest has : %.i"%len(Xtest_columns)+" columns"
```

     Xtest has : 77 columns



```python
Xtrain_columns= Xtrain.columns
print "Xtrain has : %.i"%len(Xtrain_columns)+" columns"
```

    Xtrain has : 91 columns


* Evidently Xtest and Xtrain have DIFFERENT columns and needs to be cleaned (see above)


```python
# MISSING COLUMNS
misscols= []
for i in Xtrain_columns:
    if i not in Xtest_columns:
        misscols.append(i)
print "Need to add : %.i"%len(misscols)+" columns\n"       
print "*** Missing columns that need to be added to Xtest ***\n"
print misscols

##FLEXIBILITY FOR MISSING COLUMNS- NEED TO ZERO OUT VALUES
#ADD TO XTEST DF AS ZERO VALUES
if len(misscols)>0:
    for i in misscols:
        Xtest[i]= 0
```

    Need to add : 25 columns
    
    *** Missing columns that need to be added to Xtest ***
    
    ['location_East Hanover, NJ', 'location_Jamaica, NY', 'location_Montvale, NJ', u'company_Access Staffing LLC', u'company_Aequor Technologies', u'company_Biocair, Inc', u'company_DEPARTMENT OF FINANCE', u'company_DEPT OF ENVIRONMENT PROTECTION', u'company_DEPT. OF HOMELESS SERVICES', u'company_Drexel University', u'company_Execu | Search', u'company_Icahn School of Medicine at Mount Sinai', u'company_Jobspring Partners', u'company_Mirador Real Estate, LLC', u'company_Oliver James Associates', u'company_POLICE DEPARTMENT', u'company_Pronix', u'company_Smith Hanley Associates', u'company_The Bachrach Group', u'company_UDig', u'company_Urban Scholars Program, City College of New York', u'company_VROOM', u'company_Venturi Ltd', u'company_Whiz Finder Corporation', u'company_WorldCover']



```python
Xtest.shape
```




    (32, 102)



* Ensure no 'extra' columns in the Test data, compared to Train data
    - if Test includes columsn that are NOT in Train data, delete these columns
* Similarly, if Test is MISSING columns from Train data, need to add these columns


```python
# EXTRA COLUMNS
extracols=[]
for i in Xtest_columns:
    if i not in Xtrain_columns:
        extracols.append(i)
print "Xtest Extra columns to be deleted : %.i"%(len(extracols))
print "*** DELETE THESE EXTRA COLUMNS*** \n"
print extracols



#DROP EXISTING COLUMNS INPLACE IN XTEST
for i in extracols:
    Xtest.drop(i, axis=1, inplace=True)
```

    Xtest Extra columns to be deleted : 11
    *** DELETE THESE EXTRA COLUMNS*** 
    
    ['location_Whippany, NJ', u"company_ADMIN FOR CHILDREN'S SVCS", u'company_DEPT OF PARKS & RECREATION', u'company_DemystData', u'company_Kennedy Unlimited Inc, Professional Staffing', u'company_Laguna Source', u'company_PMES', u'company_Princeton Consulting', u'company_Recruiting Resources Co.', u'company_S.C. International', u'company_Vice Media Inc.']



```python
Xtest.shape
```




    (32, 91)



## To summarize:
#### XTrain has 91 columns (64 rows), XTest should match the columns
* XTest initial= 77 columns (39rows)
* ADDED 25 
* DELETED 11  
####  FINAL Xtest = 91 columns (32 rows), PERFECT!

### Ensure Order of Columns also matches


```python
Xtrain.shape
```




    (64, 91)




```python
# DEFINE COLUMN ORDER TO MATCH XTRAIN.COLUMNS
col_order= Xtrain.columns

# MAKE XTEST THE SAME AS XTRAIN USING COL_ORDER
Xtest= Xtest[col_order]
```

### Create Binomial Salary Classification using Mean Salary from TestData set
* Mean salary from training data set is $107,000
* HIGH salary > MEDIAN
* LOW salary < MEDIAN


```python
## CREATE SALARY CLASSIFICATION
# BINOMIAL- HIGH / LOW
# USE MEDIAN SALARY AS THE BIFURCATION
# ALREADY CONFIRMED NO NULL VALUES FOR SALARY IN TESTDATA 
def class_salary(salary):
    if salary >= y_train.mean():
        return 1
    else:
        return 0
```


```python
y_train.mean()
```




    107113.875




```python
#redfine the 'y' series for both Train and Test sets with the binomial classification
y_test =y_test.apply(class_salary)
y_train=y_train.apply(class_salary)
```

### KNN Classification


```python
#KNN classification, testing across range of K values 
from sklearn.model_selection import cross_val_score
best_k = 0
best_score = 0
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain, y_train)
    predictions = knn.predict(Xtest)
    acc = accuracy_score(y_test,predictions)
    if acc > best_score:
        best_score = acc
        best_k = k

print best_k, best_score
```

    5 0.8125


* Best result is K=5, five neighbors 
* Predicted with accuracy score of nearly 80%

# REGRESSIONS

## Regression1:  L2 


```python
##Logistic Regression, Default is L2
logit = linear_model.LogisticRegressionCV(Cs=100, cv=3)
model= logit.fit(Xtrain, y_train)
pred= model.predict(Xtest)
cm= confusion_matrix(y_test, pred, labels=logit.classes_)
cm= pd.DataFrame(cm, columns=logit.classes_, index=logit.classes_)
print cm
```

        0   1
    0  11   3
    1   4  14


#### Classification Report


```python
print 'Optimal C was: ' + str(logit.C_)
print 'Intercept: ' + str(logit.intercept_)
```

    Optimal C was: [ 0.06734151]
    Intercept: [-0.46799119]



```python
print(classification_report(y_test, pred, labels=logit.classes_))
```

                 precision    recall  f1-score   support
    
              0       0.73      0.79      0.76        14
              1       0.82      0.78      0.80        18
    
    avg / total       0.78      0.78      0.78        32
    


* Confusion Matrix shows Recall (TPR= TP/P= TP/(TP+FN)= 11/(11+3) = 11/14= 79% for LOW SALARY and 78% HIGH SALARY
* PPR/Precision 73% for LOW and 82% for HIGH

#### PREDICTIONS


```python
# Predict salary_class 1= High salary (above median), 0=low salary
logit.predict(Xtest)
```




    array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
           0, 0, 0, 1, 1, 1, 0, 1, 1])




```python
# Predict probabilities
logit.predict_proba(Xtest)
```




    array([[ 0.39246027,  0.60753973],
           [ 0.62759056,  0.37240944],
           [ 0.5748641 ,  0.4251359 ],
           [ 0.31431869,  0.68568131],
           [ 0.43190827,  0.56809173],
           [ 0.45892691,  0.54107309],
           [ 0.31431869,  0.68568131],
           [ 0.62409279,  0.37590721],
           [ 0.58173299,  0.41826701],
           [ 0.46207656,  0.53792344],
           [ 0.69255288,  0.30744712],
           [ 0.45204264,  0.54795736],
           [ 0.35857262,  0.64142738],
           [ 0.52772697,  0.47227303],
           [ 0.58160443,  0.41839557],
           [ 0.49416948,  0.50583052],
           [ 0.70046146,  0.29953854],
           [ 0.47310058,  0.52689942],
           [ 0.54612483,  0.45387517],
           [ 0.51903752,  0.48096248],
           [ 0.36246068,  0.63753932],
           [ 0.39246027,  0.60753973],
           [ 0.51903752,  0.48096248],
           [ 0.51903752,  0.48096248],
           [ 0.53693429,  0.46306571],
           [ 0.58219126,  0.41780874],
           [ 0.40127369,  0.59872631],
           [ 0.38040449,  0.61959551],
           [ 0.43190827,  0.56809173],
           [ 0.69621214,  0.30378786],
           [ 0.36604699,  0.63395301],
           [ 0.38765656,  0.61234344]])




```python
def examine_coefficients(model, df):
    df = pd.DataFrame(
        { 'Coefficient' : model.coef_[0] , 'Feature' : df.columns}
    ).sort_values(by='Coefficient')
    return df[df.Coefficient !=0 ]
```


```python
logit_ex_coef= examine_coefficients(model, Xtest)
logit_ex_coef
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52</th>
      <td>-0.262163</td>
      <td>location_Manhattan, NY</td>
    </tr>
    <tr>
      <th>66</th>
      <td>-0.103759</td>
      <td>company_DEPT OF HEALTH/MENTAL HYGIENE</td>
    </tr>
    <tr>
      <th>39</th>
      <td>-0.088316</td>
      <td>research</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.082671</td>
      <td>analysis</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.078705</td>
      <td>manager</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.071985</td>
      <td>coordinator</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-0.061818</td>
      <td>company_POLICE DEPARTMENT</td>
    </tr>
    <tr>
      <th>85</th>
      <td>-0.059540</td>
      <td>company_Urban Scholars Program, City College o...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.059540</td>
      <td>instructor</td>
    </tr>
    <tr>
      <th>47</th>
      <td>-0.057201</td>
      <td>technician</td>
    </tr>
    <tr>
      <th>62</th>
      <td>-0.055442</td>
      <td>company_Columbia University</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.054488</td>
      <td>fellow</td>
    </tr>
    <tr>
      <th>34</th>
      <td>-0.054488</td>
      <td>postdoctoral</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-0.054488</td>
      <td>company_Icahn School of Medicine at Mount Sinai</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.049954</td>
      <td>bureau</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.049888</td>
      <td>assistant</td>
    </tr>
    <tr>
      <th>50</th>
      <td>-0.049353</td>
      <td>location_East Hanover, NJ</td>
    </tr>
    <tr>
      <th>88</th>
      <td>-0.049353</td>
      <td>company_Whiz Finder Corporation</td>
    </tr>
    <tr>
      <th>63</th>
      <td>-0.048593</td>
      <td>company_DEPARTMENT OF FINANCE</td>
    </tr>
    <tr>
      <th>67</th>
      <td>-0.041232</td>
      <td>company_DEPT. OF HOMELESS SERVICES</td>
    </tr>
    <tr>
      <th>89</th>
      <td>-0.039240</td>
      <td>company_WorldCover</td>
    </tr>
    <tr>
      <th>82</th>
      <td>-0.039240</td>
      <td>company_Smith Hanley Associates</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.035651</td>
      <td>computer</td>
    </tr>
    <tr>
      <th>32</th>
      <td>-0.034120</td>
      <td>panel</td>
    </tr>
    <tr>
      <th>38</th>
      <td>-0.034120</td>
      <td>registr</td>
    </tr>
    <tr>
      <th>49</th>
      <td>-0.034120</td>
      <td>world</td>
    </tr>
    <tr>
      <th>48</th>
      <td>-0.034120</td>
      <td>trade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.034120</td>
      <td>center</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-0.034120</td>
      <td>maintenance</td>
    </tr>
    <tr>
      <th>84</th>
      <td>-0.033827</td>
      <td>company_UDig</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.034581</td>
      <td>company_Access Staffing LLC</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.034581</td>
      <td>company_Execu | Search</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.037415</td>
      <td>senior</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.040230</td>
      <td>affairs</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.040230</td>
      <td>company_Aequor Technologies</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.045313</td>
      <td>media</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.051608</td>
      <td>nlp</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.056276</td>
      <td>hadoop</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.061777</td>
      <td>dollar</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.061777</td>
      <td>driven</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.061777</td>
      <td>multi</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.061777</td>
      <td>million</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.063085</td>
      <td>finance</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.063085</td>
      <td>sr</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.069202</td>
      <td>company</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.070806</td>
      <td>company_Venturi Ltd</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.070806</td>
      <td>fintech</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.073416</td>
      <td>learning</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.073416</td>
      <td>machine</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.074561</td>
      <td>python</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.075515</td>
      <td>big</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.089109</td>
      <td>company_Enterprise Select</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.104732</td>
      <td>director</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.111998</td>
      <td>engineer</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.164187</td>
      <td>quantitative</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.171011</td>
      <td>company_Harnham</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.191561</td>
      <td>company_Averity</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.258568</td>
      <td>data</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.265437</td>
      <td>location_New York, NY</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.337591</td>
      <td>scientist</td>
    </tr>
  </tbody>
</table>
<p>91 rows × 2 columns</p>
</div>




```python
len(logit_ex_coef)
```




    91




```python
## MAX POSTIIVE COEFFICIENT 
logit_ex_coef.loc[logit_ex_coef['Coefficient']== logit_ex_coef['Coefficient'].max()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>0.337591</td>
      <td>scientist</td>
    </tr>
  </tbody>
</table>
</div>




```python
## MIN  COEFFICIENT 
logit_ex_coef.loc[logit_ex_coef['Coefficient']== logit_ex_coef['Coefficient'].min()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52</th>
      <td>-0.262163</td>
      <td>location_Manhattan, NY</td>
    </tr>
  </tbody>
</table>
</div>



* Interestingly Manhattan has a NEGATIVE effect on salary! It's the most highly negative coefficient!
* Scientist, 'New York, NY', data, engineer, director all played strong positive coefficients
* Limited data set definitely creating these 'interesting' outcomes and should be mitigated with more samples, additional cities and further variable selection
* 91 variables used in the default (L2) regression

## LOGISTIC REGRESSION - USING L1 


```python
##Logistic Regression, L1 penalty
l1_logit = linear_model.LogisticRegressionCV(penalty= 'l1', Cs=100, cv=3, solver='liblinear')
l1_model= l1_logit.fit(Xtrain, y_train)
l1_pred= l1_model.predict(Xtest)
l1_cm= confusion_matrix(y_test, l1_pred, labels=l1_logit.classes_)
l1_cm= pd.DataFrame(l1_cm, columns=l1_logit.classes_, index=l1_logit.classes_)
print l1_cm
```

        0   1
    0  10   4
    1   4  14


### Classification Report


```python
print 'Optimal C was: ' + str(l1_logit.C_)
print 'Intercept: ' + str(l1_logit.intercept_)
```

    Optimal C was: [ 2257.01971963]
    Intercept: [-0.44970065]



```python
print(classification_report(y_test, l1_pred, labels=l1_logit.classes_))
```

                 precision    recall  f1-score   support
    
              0       0.71      0.71      0.71        14
              1       0.78      0.78      0.78        18
    
    avg / total       0.75      0.75      0.75        32
    


* L1 penalty logit regression actually worsened the efficacy of the model

### Predictions


```python
# Predict salary_class 1= High salary (above median), 0=low salary
l1_logit.predict(Xtest)
```




    array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
           0, 0, 0, 1, 1, 1, 0, 1, 1])




```python
# Predict probabilities
l1_logit.predict_proba(Xtest)
```




    array([[  3.52292253e-04,   9.99647708e-01],
           [  9.99916559e-01,   8.34408112e-05],
           [  9.99907402e-01,   9.25978961e-05],
           [  1.16042923e-04,   9.99883957e-01],
           [  1.11200226e-04,   9.99888800e-01],
           [  5.20188253e-06,   9.99994798e-01],
           [  1.16042923e-04,   9.99883957e-01],
           [  7.30955685e-01,   2.69044315e-01],
           [  9.99907402e-01,   9.25978961e-05],
           [  2.30665607e-02,   9.76933439e-01],
           [  9.99999993e-01,   6.75289493e-09],
           [  2.58404316e-04,   9.99741596e-01],
           [  1.19990236e-04,   9.99880010e-01],
           [  8.90546596e-03,   9.91094534e-01],
           [  9.99939515e-01,   6.04854330e-05],
           [  4.90955344e-07,   9.99999509e-01],
           [  9.99999993e-01,   6.75289493e-09],
           [  2.67209597e-03,   9.97327904e-01],
           [  9.99546011e-01,   4.53988964e-04],
           [  8.11241630e-01,   1.88758370e-01],
           [  5.73955662e-05,   9.99942604e-01],
           [  4.53853989e-04,   9.99546146e-01],
           [  8.11241630e-01,   1.88758370e-01],
           [  8.11241630e-01,   1.88758370e-01],
           [  6.52692780e-01,   3.47307220e-01],
           [  9.99965094e-01,   3.49057671e-05],
           [  3.54090052e-04,   9.99645910e-01],
           [  2.67209597e-03,   9.97327904e-01],
           [  1.11200226e-04,   9.99888800e-01],
           [  9.99999993e-01,   6.75289493e-09],
           [  1.19990236e-04,   9.99880010e-01],
           [  2.67209597e-03,   9.97327904e-01]])




```python
l1_ex_coef= examine_coefficients(l1_model, Xtest)
l1_ex_coef
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>-13.601053</td>
      <td>company_WorldCover</td>
    </tr>
    <tr>
      <th>82</th>
      <td>-13.589164</td>
      <td>company_Smith Hanley Associates</td>
    </tr>
    <tr>
      <th>52</th>
      <td>-9.526143</td>
      <td>location_Manhattan, NY</td>
    </tr>
    <tr>
      <th>39</th>
      <td>-8.221495</td>
      <td>research</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-7.851152</td>
      <td>company_POLICE DEPARTMENT</td>
    </tr>
    <tr>
      <th>84</th>
      <td>-7.834332</td>
      <td>company_UDig</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-7.795599</td>
      <td>manager</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-6.897431</td>
      <td>coordinator</td>
    </tr>
    <tr>
      <th>47</th>
      <td>-6.879980</td>
      <td>technician</td>
    </tr>
    <tr>
      <th>88</th>
      <td>-5.015027</td>
      <td>company_Whiz Finder Corporation</td>
    </tr>
    <tr>
      <th>85</th>
      <td>-4.559344</td>
      <td>company_Urban Scholars Program, City College o...</td>
    </tr>
    <tr>
      <th>64</th>
      <td>-4.093057</td>
      <td>company_DEPARTMENT OF TRANSPORTATION</td>
    </tr>
    <tr>
      <th>65</th>
      <td>-3.324007</td>
      <td>company_DEPT OF ENVIRONMENT PROTECTION</td>
    </tr>
    <tr>
      <th>50</th>
      <td>-2.949706</td>
      <td>location_East Hanover, NJ</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-2.918259</td>
      <td>company_Icahn School of Medicine at Mount Sinai</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.629681</td>
      <td>analysis</td>
    </tr>
    <tr>
      <th>34</th>
      <td>-2.346234</td>
      <td>postdoctoral</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-1.897247</td>
      <td>instructor</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-1.650413</td>
      <td>fellow</td>
    </tr>
    <tr>
      <th>90</th>
      <td>-1.041852</td>
      <td>state_ NY</td>
    </tr>
    <tr>
      <th>66</th>
      <td>-0.692273</td>
      <td>company_DEPT OF HEALTH/MENTAL HYGIENE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.549776</td>
      <td>analyst</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.073477</td>
      <td>bureau</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.033454</td>
      <td>learning</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.490005</td>
      <td>hadoop</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1.775065</td>
      <td>company_Analytic Recruiting</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2.019648</td>
      <td>company_Oliver James Associates</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2.023389</td>
      <td>company_ERS Search</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2.028481</td>
      <td>company_All-In Analytics</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2.052556</td>
      <td>company_Jobspring Partners</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2.176182</td>
      <td>data</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2.215740</td>
      <td>company_Enterprise Select</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2.379675</td>
      <td>company_VROOM</td>
    </tr>
    <tr>
      <th>74</th>
      <td>3.105764</td>
      <td>company_Harnham</td>
    </tr>
    <tr>
      <th>36</th>
      <td>3.515128</td>
      <td>python</td>
    </tr>
    <tr>
      <th>56</th>
      <td>3.600177</td>
      <td>company_Aequor Technologies</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3.843270</td>
      <td>company_Averity</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4.570690</td>
      <td>affairs</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4.650313</td>
      <td>director</td>
    </tr>
    <tr>
      <th>41</th>
      <td>5.237587</td>
      <td>scientist</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6.203697</td>
      <td>engineer</td>
    </tr>
    <tr>
      <th>37</th>
      <td>10.364544</td>
      <td>quantitative</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(l1_ex_coef)
```




    42




```python
## MAX POSTIIVE COEFFICIENT 
l1_ex_coef.loc[l1_ex_coef['Coefficient']== l1_ex_coef['Coefficient'].max()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>10.364544</td>
      <td>quantitative</td>
    </tr>
  </tbody>
</table>
</div>




```python
## MIN COEFFICIENT 
l1_ex_coef.loc[l1_ex_coef['Coefficient']== l1_ex_coef['Coefficient'].min()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>-13.601053</td>
      <td>company_WorldCover</td>
    </tr>
  </tbody>
</table>
</div>



### IF I HAD MORE TIME:
#### Take only companies with count of positions > 10, make rest other
#### add industry variable
#### clean location further (eg. United States) 
#### add additional states/areas


```python

```
