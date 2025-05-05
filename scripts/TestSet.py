import pandas as pd

dic = {}

queries = ["I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.", "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options", "I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour", "Content Writer required, expert in English and SEO.",
    """
    Find me 1 hour long assesment for
the below job at SHL
Job Description
Join a community that is shaping
the future of work! SHL, People
Science. People Answers.
Are you a seasoned QA Engineer
with a flair for innovation? Are you
ready to shape the future of talent
assessment and empower
organizations to unlock their full
potential? If so, we want you to be a
part of the SHL Team! As a QA
Engineer, you will be involved in
creating and implementing software
solutions that contribute to the
development of our groundbreaking products.
An excellent benefit package is
offered in a culture where career
development, with ongoing
manager guidance, collaboration,
flexibility, diversity, and inclusivity
are all intrinsic to our culture. There
is a huge investment in SHL
currently so there’s no better time
to become a part of something
transformational.
What You Will Be Doing
Getting involved in engineering
quality assurance and providing
inputs when required.
Create and develop test plans for
various forms of testing.
Conducts and/or participates in
formal and informal test case
reviews.
Develop and initiate functional
tests and regression tests.
Rolling out improvements for
testing and quality processes.
Essential
What we are looking for from you:
Development experience – Java or
JavaScript, CSS, HTML (Automation)
Selenium WebDriver and page
object design pattern (Automation)
SQL server knowledge
Test case management experience.
Manual Testing
Desirable
Knowledge the basic concepts of
testing
Strong solution-finding experience
Strong verbal and written
communicator.
Get In Touch
Find out how this one-off
opportunity can help you achieve
your career goals by making an
application to our knowledgeable
and friendly Talent Acquisition
team. Choose a new path with SHL.
#CareersAtSHL #SHLHiringTalent
#TechnologyJobs
#QualityAssuranceJobs
#CareerOpportunities
#JobOpportunities
About Us
We unlock the possibilities of
businesses through the power of
people, science and technology.
We started this industry of people
insight more than 40 years ago and
continue to lead the market with
powerhouse product launches,
ground-breaking science and
business transformation.
When you inspire and transform
people’s lives, you will experience
the greatest business outcomes
possible. SHL’s products insights,
experiences, and services can help
achieve growth at scale. 
What SHL Can Offer You
Diversity, equity, inclusion and
accessibility are key threads in the
fabric of SHL’s business and culture
(find out more about DEI and
accessibility at SHL )
Employee benefits package that
takes care of you and your family.
Support, coaching, and on-the-job
development to achieve career
success
A fun and flexible workplace where
you’ll be inspired to do your best
work (find out more LifeAtSHL )
The ability to transform workplaces
around the world for others.
SHL is an equal opportunity
employer. We support and
encourage applications from a
diverse range of candidates. We
can, and do make adjustments to
make sure our recruitment process
is as inclusive as possible.
SHL is an equal opportunity
employer.
    """,
    
    "ICICI Bank Assistant Admin, Experience required 0-2 years, test should be 30-40 mins long", 
    
    """"
    KEY RESPONSIBITILES:
Manage the sound-scape of the station through appropriate
creative and marketing
interventions to Increase or
Maintain the listenership
Acts as an interface between
Programming & sales team, thereby
supporting the sales team by
providing creative inputs in order to
increase the overall ad spends by
clients
Build brand Mirchi by ideating fresh
programming initiatives on air
campaigns, programming led onground events & new properties to
ensure brand differentiation & thus
increase brand recall at station level
Invest time in local RJs to grow &
develop them as local celebrities
Through strong networking, must
focus on identifying the best of local
talent and ensure to bring the
creative minds from the market on
board with Mirchi
Build radio as a category for both
listeners & advertisers
People Management
Identifying the right talent and
investing time in developing them
by frequent feedback on their
performance
Monitor, Coach and mentor team
members on a regular basis
Development of Jocks as per
guidelines
Must have an eye to spot the local
talent to fill up vacancies locally
TECHNICAL SKILLS &
QUALIFICATION REQUIRED:
Graduation / Post Graduation (Any
specialisation) with 8 -12 years of
relevant experience
Experience in digital content
conceptualisation
Strong branding focus
Must be well-read in variety of
areas and must keep up with the
latest events in the city / cluster /
country
Must know to read, write & speak
English
PERSONAL ATTRIBUTES:
Excellent communication skills
Good interpersonal skills
People management
Suggest me some tests for the
above jd. The duration should be at
most 90 mins
    """]

print(queries[-1])

outs = [
    ["https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/", "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/", "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/", "https://www.shl.com/solutions/products/product-catalog/view/core-java-advanced-level-new/", "https://www.shl.com/solutions/products/product-catalog/view/agile-software-development/", "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-8-0-job-focused-assessment/", "https://www.shl.com/solutions/products/product-catalog/view/computer-science-new/"],
    
    ["https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-7-1/", "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-sift-out-7-1/", "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-solution/", "https://www.shl.com/solutions/products/product-catalog/view/sales-representative-solution/", "https://www.shl.com/solutions/products/product-catalog/view/sales-support-specialist-solution/", "https://www.shl.com/solutions/products/product-catalog/view/technical-sales-associate-solution/", "https://www.shl.com/solutions/products/product-catalog/view/svar-spoken-english-indian-accent-new/", "https://www.shl.com/solutions/products/product-catalog/view/sales-and-service-phone-solution/", "https://www.shl.com/solutions/products/product-catalog/view/sales-and-service-phone-simulation/", "https://www.shl.com/solutions/products/product-catalog/view/english-comprehension-new/"],
    
    ["https://www.shl.com/solutions/products/product-catalog/view/motivation-questionnaire-mqm5/", "https://www.shl.com/solutions/products/product-catalog/view/global-skills-assessment/", "https://www.shl.com/solutions/products/product-catalog/view/graduate-8-0-job-focused-assessment-4228/"],

    ["https://www.shl.com/solutions/products/product-catalog/view/drupal-new/", "https://www.shl.com/solutions/products/product-catalog/view/search-engine-optimization-new/", "https://www.shl.com/solutions/products/product-catalog/view/administrative-professional-short-form/", "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-sift-out-7-1/", "https://www.shl.com/solutions/products/product-catalog/view/general-entry-level-data-entry-7-0-solution/"],

    ["https://www.shl.com/solutions/products/product-catalog/view/automata-selenium/", "https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/", "https://www.shl.com/solutions/products/product-catalog/view/automata-front-end/", "https://www.shl.com/solutions/products/product-catalog/view/javascript-new/", "https://www.shl.com/solutions/products/product-catalog/view/htmlcss-new/", "https://www.shl.com/solutions/products/product-catalog/view/html5-new/", "https://www.shl.com/solutions/products/product-catalog/view/css3-new/", "https://www.shl.com/solutions/products/product-catalog/view/selenium-new/", "https://www.shl.com/solutions/products/product-catalog/view/sql-server-new/", "https://www.shl.com/solutions/products/product-catalog/view/automata-sql-new/", "https://www.shl.com/solutions/products/product-catalog/view/manual-testing-new/"],

    ["https://www.shl.com/solutions/products/product-catalog/view/administrative-professional-short-form/", "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-ability/", "https://www.shl.com/solutions/products/product-catalog/view/financial-professional-short-form/", "https://www.shl.com/solutions/products/product-catalog/view/bank-administrative-assistant-short-form/", "https://www.shl.com/solutions/products/product-catalog/view/general-entry-level-data-entry-7-0-solution/", "https://www.shl.com/solutions/products/product-catalog/view/basic-computer-literacy-windows-10-new/"],

    ["https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-ability-next-generation/", "https://www.shl.com/solutions/products/product-catalog/view/shl-verify-interactive-inductive-reasoning/", "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32r/"]
]

