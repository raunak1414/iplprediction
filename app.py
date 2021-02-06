from flask import Flask,render_template,request,redirect,url_for
import pickle as pkl
import numpy as np
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    team1=str(request.args.get('team1'))
    team2=str(request.args.get('team2'))

    if team1==team2:
        return redirect(url_for('index'))

    toss_winner=int(request.args.get('toss_winner'))
    choose=int(request.args.get('toss_decision'))

    with open('model.pkl','rb') as f:
        model=pkl.load(f)
        
    with open('inv_vocab.pkl','rb') as f:
        inv_vocab=pkl.load(f)

    cteam1=inv_vocab[team1]
    cteam2=inv_vocab[team2]

    arr=np.array([cteam1,cteam2,choose,toss_winner]).reshape(1,-1)
    predict=model.predict(arr)

    if(predict==0):
        return render_template('after.html',data=team1)
    else:
        return render_template('after.html',data=team2)

    

    return render_template('after.html',data=predict)

if __name__=="__main__":
    app.run(debug=True)