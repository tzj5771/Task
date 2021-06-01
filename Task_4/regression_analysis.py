from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from Task_3.data_load_handle import DataLoadHandle


class RegressionAnalysis(DataLoadHandle):
    def __init__(self):
        super().__init__()

    def run(self):
        # 第1步：导入线性回归
        from sklearn.linear_model import LinearRegression
        # 第2步：创建模型：线性回归
        model = LinearRegression()
        tt = TfidfVectorizer(max_df=0.5)
        tf = tt.fit_transform(self.train_documents)
        # 第3步：训练模型
        model.fit(tf, self.train_labels)
        # 截距
        a = model.intercept_
        # 回归系数
        b = model.coef_
        print('最佳拟合线：截距a=', a, '，回归系数b=', b)
        # 绘图
        import matplotlib.pyplot as plt
        # 训练数据散点图
        plt.scatter(self.train_documents, self.train_labels, color='blue', label="train data")
        # 训练数据的预测值
        y_train_pred = model.predict(self.train_documents)
        # 绘制最佳拟合线
        plt.plot(self.train_documents, y_train_pred, color='black', linewidth=3, label="best line")
        # 添加图标标签
        plt.legend(loc=2)
        plt.xlabel("Hours")
        plt.ylabel("Score")
        # 显示图像
        plt.show()

if __name__ == '__main__':
    ra = RegressionAnalysis()
    print(ra.run())
