Hi there! Thanks for coming to check out my project :D Below is a couple of FAQs and the current progress:

<h3>1. What is the goal of this project?</h3>
<ul>The goal is to take records of diabetic patients in the US, and find the most important features in predicting readmission within 30 days by training a ML model.</ul>

<h3>1a. But wait, wouldn't the doctor be able to gauge what their patient will need?</h3>
<ul>Of couse best case scenario is your doctor being able to reach the conclusion that you will need close monitoring and you are high risk for readmission.

But what happens if the doctor doesn't have the time or resources to sit down and communicate to assess risk? What if there are gaps in your medical records because of migrating providers or your insurance company is not dilligent enough?

In those cases, for your doctor or discharge nurse, being able to necessary data and get an accurate gauge in if the patient needs follow up or close monitoring would help significantly in the  quality and long term outcome of the care provided to the patient.</ul>

<h3>2. What can we expect from this project?</h3>
<ul>The final product will be an interactive dashboard where you can input your own patient data and get a probability predicting readmission within 30 days.</ul>

<h3>3. What datasets are being used?</h3>
<ul><a href="https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008">Diabetic patient records dataset</a> courtesy of Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore</ul>
<ul><a href="https://data.cms.gov/provider-data/topics/hospitals">Hospital quality data dataset</a> courtesy of data.cms.gov</ul>

<h3>Current progress! Now loading . . . .</h3>
<p> 31/5/2026 - datasets have been loaded, .gitignore has been fixed, and here are the current insights (copied from markdown in the notebook):
<h4>## Baseline Readmission Rate<h4>
<ul>After examining the Excel output, each category is always around 13% for the readmission rates regardless of the value.
So a model predicting binary value (readmitted within or more than 30 days) would achieve 87% accuracy as well.
Therefore F1 score and recall will be the main measurment of acurracy for our model.<ul>

<h4>## Readmission Rate Distributions<h4>
<ul>Number of inpatient visits might be the strongest indicator since as this number increases, the more likely they will be readmitted because it is indictor of severity or the chronic level of diabetes.
Contrary to this, time spent in hospital does not visibly guard against readmission since the rate stays relatively stable as time spent increases. <ul>
