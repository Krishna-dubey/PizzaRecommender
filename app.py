import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import os.path
from rivescript import RiveScript

rive = os.path.dirname(__file__)
bot = RiveScript()
bot.load_directory(rive)
bot.sort_replies()


def getMyPizza(ingredients):
    pizza_df = pd.read_csv("Pizza Ingredients.csv",na_values=['?'," ",""])
    pizza_df.Ingredients.replace(to_replace="[|]",value=" ",inplace=True,regex=True)
    myRow = ['MyPizza']
    myRow.append(ingredients)
    pizza_df.loc[len(pizza_df)] = myRow

    cv=CountVectorizer()
    cv_matrix=cv.fit_transform(pizza_df['Ingredients']) #gives the matrix of n*n with count of words matched
    cs=cosine_similarity(cv_matrix) #gives cosine similarity

    table=pd.DataFrame(index=pizza_df.PizzaName,columns=pizza_df.PizzaName,data=cs)
    table=table.drop(['MyPizza'],axis=1)
    if str(max(table.loc["MyPizza"])) != '0.0':
        return table.idxmax(axis=1)['MyPizza']
    else:
        return 0

def getReply(ingredients):
    ans = getMyPizza(ingredients)
    if ans != 0:
        return "You may like: "+ans
    else:
        return str(bot.reply('localuser',ingredients)) 
    

app = Flask(__name__)

@app.route("/")
def home():    
    return render_template("home.html") 
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    return getReply(userText)

if __name__ == "__main__":    
    app.run(debug=True)



