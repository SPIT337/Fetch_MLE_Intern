import base64
import io
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open('static/arima.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    if request.method == 'POST':
        n = int(request.form['input'])
        vals = model.predict(n)
        dates = pd.date_range(start='2022-01-01', periods=n)
        df = pd.DataFrame({'date': dates, 'value': vals})
        plt.switch_backend('Agg') 

        fig, ax = plt.subplots()
        ax.plot_date(df['date'], df['value'], '-')
        ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        ax.grid(which='both')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', plot_url=plot_url)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
