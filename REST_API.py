import pickle
from flask import Flask, jsonify, json, request
import numpy as np

app = Flask(__name__)

# --- the filename for the model ---
filename = 'diebetes.sav'

# --- load the saved model ---
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/diebetes/predict', methods=['POST'])

def predict():
	# --- get the features tp predict ---
	features = request.json

	# create the features
	features_list = [features['Glucose'],
					features['BMI'],
					features['Age']
						]

	# --- get the prediction class ---
	prediction = loaded_model.predict([features_list])
	if prediction == 1:
		prediction = 'Diebetic'
	else:
		prediction = 'Non-Diebetic'

	# --- get the prediction probabilities
	confidence = loaded_model.predict_proba([features_list])

	# --- formulate the response to return to the client
	response = {}
	response['prediction'] = str(prediction)
	response['confidence'] = str(round(np.amax(confidence[0]) * 100, 2))

	return jsonify(response)


if __name__ == '__main__':
	app.run(debug=True, port=5000)