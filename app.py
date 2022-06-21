from flask import Flask, render_template, request
import joblib
import torch
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as Fun

app = Flask(__name__)
print(joblib.__version__)

@app.route('/')
def Home():
   return render_template('HOME.html')

@app.route('/Quartz',methods = ['POST', 'GET'])
def Quartz():
   return render_template('Quartz.html')

@app.route('/Quartz/A',methods = ['POST', 'GET'])
def A():
   return render_template('index2.html')

@app.route('/Quartz/B',methods = ['POST', 'GET'])
def B():
   return render_template('index.html')
@app.route('/Quartz/A/model1',methods = ['POST'])
def result():
    class Net(torch.nn.Module):
      def __init__(self, n_feature, n_hidden, n_output):
         super(Net, self).__init__()
         self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐藏层网络
         self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
         self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
         self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
         self.hidden5 = torch.nn.Linear(n_hidden, n_hidden)
         self.hidden6 = torch.nn.Linear(n_hidden, n_hidden)
         self.hidden7 = torch.nn.Linear(n_hidden, n_hidden)
         self.out = torch.nn.Linear(n_hidden, n_output)  # 定义输出层网络

      def forward(self, x):
         x = Fun.tanh(self.hidden(x))  # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
         x = Fun.tanh(self.hidden2(x))
         x = Fun.tanh(self.hidden3(x))
         x = Fun.dropout(x, p=0.3)
         x = Fun.tanh(self.hidden4(x))
         x = Fun.tanh(self.hidden5(x))
         x = Fun.dropout(x, p=0.3)
         x = Fun.tanh(self.hidden6(x))
         x = Fun.tanh(self.hidden7(x))
         x = Fun.dropout(x, p=0.3)
         x = self.out(x)  # 输出层不用激活函数
         return x

    model = Net(n_feature=5, n_hidden=100, n_output=9)
    model.load_state_dict(torch.load('Classifier1.pth'))
    X = np.loadtxt('Quartzdataset.csv', delimiter=",", usecols=(1, 2, 3, 4, 5), dtype=float, skiprows=1)
    dtype = ['IRG', 'carlin', 'epithermal', 'granite', 'greisen', 'orogenic', 'pegmatite', 'porphyry', 'skarn']
    a = []
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features, dtype=float)
    X = np.row_stack((X, features))
    X = np.log(X + 1)
    X = preprocessing.scale(X)  # x是要进行标准化的样本数据
    features = X[-1, :]
    a.append(features)
    a = np.array(a, float)
    a = torch.FloatTensor(a)
    outputs = model(a)
    prediction = torch.max(outputs, 1)[1]
    prediction = prediction.data.numpy()
    prediction = dtype[prediction[0]]
    return render_template('predict_result.html', prediction_text=prediction)

@app.route('/Quartz/B/model2',methods = ['POST'])
def result2():
    X = np.loadtxt('Quartzdataset.csv', delimiter=",", usecols=(1, 2, 3, 4, 5), dtype=float,
                       skiprows=1)  # 返回数据集所有输入特征
    a=[]
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features, dtype=float)
    X = np.row_stack((X, features))
    X = np.log(X + 1)
    X = preprocessing.scale(X)
    features = X[-1, :]
    a.append(features)
    a = np.array(a)
    model1 = joblib.load('SVM.pkl')
    prediction = model1.predict(a)
    prediction=prediction[0]
    return render_template("predict_result.html", prediction_text=prediction)


if __name__ == '__main__':
   app.run(debug = True)