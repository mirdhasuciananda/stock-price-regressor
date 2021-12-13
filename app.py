from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('model.h5')
vars = 'vars.pkl'

def generate_plot(model, vars, n_days):
    with open('vars.pkl', 'rb') as f:
        scaler = pickle.load(f)
        X_test = pickle.load(f)
        df_price = pickle.load(f)

    y_pred_lstm = model.predict(X_test)
    y_pred_lstm_unscaled = scaler.inverse_transform(y_pred_lstm)
    df_price_pred = pd.DataFrame.from_dict({'PGAS': y_pred_lstm_unscaled[:, 0],
                                            'BBCA': y_pred_lstm_unscaled[:, 1],
                                            'UNTR': y_pred_lstm_unscaled[:, 2]})
    df_price_train = df_price.iloc[:650]

    df_result = pd.concat([df_price_train, df_price_pred], axis=0)
    df_result.reset_index(inplace=True)
    df_result.head()

    fig, ax = plt.subplots(figsize=(10, 8), nrows=3, ncols=1)
    fig.tight_layout()

    df_result['PGAS'][:650].plot(ax=ax[0], legend=True)
    df_result['PGAS'][650:650 + n_days].plot(ax=ax[0], legend=True, linewidth=2.5)
    ax[0].legend(['Training set PGAS', 'Prediction set PGAS'])
    ax[0].set_title('PGAS')

    df_result['BBCA'][:650].plot(ax=ax[1], legend=True)
    df_result['BBCA'][650:650 + n_days].plot(ax=ax[1], legend=True, linewidth=2.5)
    ax[1].legend(['Training set BBCA', 'Prediction set BBCA'])
    ax[1].set_title('BBCA')

    df_result['UNTR'][:650].plot(ax=ax[2], legend=True)
    df_result['UNTR'][650:650 + n_days].plot(ax=ax[2], legend=True, linewidth=2.5)
    ax[2].legend(['Training set UNTR', 'Prediction set UNTR'])
    ax[2].set_title('UNTR')

    # plt.show()
    plt.savefig('./static/uploads/output.png')

    return './static/uploads/output.png'


# @app.route('/')
# def index():
#     return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form:
            n_days = request.form['n_days']
            n_days = int(n_days)
            img_path = generate_plot(model, vars, n_days)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_path)
            return render_template('index.html', uploaded_image='output.png')

    return render_template('index.html')


@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
