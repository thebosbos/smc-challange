from flask import Flask, request, jsonify
import torch
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the PyTorch model
model = joblib.load('C:/Users/bouss/Desktop/smc/application/random_forest_model.joblib') 

@app.route('/predict', methods=['POST','GET'])
def predict():
    
        # # Get input data from the request
        data = request.get_json()
        
        
        values_only = [int(value) for value in data.values()]
        result_list = [values_only]
        
        # # Perform any necessary data preprocessing on input_data if needed

        # # Make predictions using the loaded PyTorch model
        prediction=model.predict(result_list)
        print(prediction)
        if prediction == 3:
            output= "votre moyyenne generale est dans l'intervalle [10 15]"
        elif prediction == 4:
            output= "votre moyyenne generale est dans l'intervalle [15 20]"
            
        print(output)
        Dépression_chronique = sum(result_list[0][:5])

# Calculate the percentage
        total_elements = 5
        percentage_depression = (Dépression_chronique / total_elements) * 100

        print("Dépression chronique :",  percentage_depression)

        Anxiété_généralisée = sum(result_list[0][6:10])

# Calculate the percentage
        total_elements_Anxiété = 5
        percentage_anxiete = (Anxiété_généralisée / total_elements_Anxiété) * 100

        print("Anxiété généralisée (TAG):", percentage_anxiete)

        TDAH_chez_adultee = sum(result_list[0][10:16])

# Calculate the percentage
        total_elements_TDAH = 6
        percentage_tdah = (TDAH_chez_adultee / total_elements_TDAH) * 100

        print("TDAH chez l’adulte:", percentage_tdah)

        # response = {'predicted_class':  prediction}
        # print(response)
        finaloutput = output+'for the first test you have a depression chronique of {} %, an anxiété généralisée (TAG) of {} %, and a TDAH chez l’adulte of {} %'.format(percentage_depression,percentage_anxiete,percentage_tdah)
        print(finaloutput)
        return jsonify({'':finaloutput})

 

if __name__ == '__main__':
    app.run(debug=True)
