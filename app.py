from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

path = './model/remote_work_model.pkl'

with open(path,'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        data = {
        'Age' : request.form.get("age"),
        'Gender' : request.form.get("gender"),
        'Job_Role' : request.form.get("Job_Role"),
        'Industry' : request.form.get("industry"),
        'Years_of_Experience' : request.form.get("experience"),
        'Work_Location' : request.form.get("work_location"),
        'Hours_Worked_Per_Week' : request.form.get("hours_worked"),
        'Number_of_Virtual_Meetings' : request.form.get("virtual_meetings"),
        'Work_Life_Balance_Rating' : request.form.get("work_life_balance"),
        'Stress_Level' : request.form.get("stress_level"),
        'Mental_Health_Condition' : request.form.get("mental_health_condition"),
        'Access_to_Mental_Health_Resources' : request.form.get("access_resources"),
        'Productivity_Change' : request.form.get("Productivity_Change"),
        'Social_Isolation_Rating' : request.form.get("isolation_rating"),
        'Company_Support_for_Remote_Work' : request.form.get("company_support"),
        'Physical_Activity' : request.form.get("physical_activity"),
        'Sleep_Quality' : request.form.get("Sleep_Quality"),
        'Region' : request.form.get("region")
        }

        # print(data)

        test_data = pd.DataFrame([data])

        predict=model.predict(test_data)
        if predict[0] == 0:
            res = "Neutral"
        elif predict[0] == 1:
            res = "Satisfied" 
        else:
            res = "Unsatisfied"     
        # print(predict)

        return render_template('result.html', res=res)

    
                                  

if "__main__" == __name__:
    app.run(debug=True)
    