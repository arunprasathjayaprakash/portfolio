#loading model and creating an endpoint
import flask
import ast
import mlflow.sklearn
import requests

model = mlflow.sklearn.load_model('models/model_V1')

app = flask.Flask(__name__)

@app.route('/',methods=['GET','POST'])
def return_predictions():
    '''
    Returns predicted probabilities from model loaded with mlflow
    
    args: request
    returns: Predicted probabilities
    '''

    try:
        params = flask.request.data

        if params:
            params = params.decode('utf-8')
            params = ast.literal_eval(params)
            data_values = params.values()

            predictions = model.predict_proba([list(data_values)])

            return flask.jsonify({"Predicted_probability":predictions[0][0]})
        else:
            return flask.abort(404,description="Please send required data")
    except Exception as e:
        return flask.abort(400,description=f"{e}")

if __name__ == "__main___":
    app.run(debug=True)