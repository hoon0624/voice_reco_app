from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/', methods=['POST'])
# def getvalue():
#     name = request.form['input']
#     return render_template('pass.html', n=name)

if __name__ == "__main__":
    app.run(debug=True)