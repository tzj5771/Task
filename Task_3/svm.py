from sklearn import tree, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from Task_3.data_load_handle import DataLoadHandle


class SVM(DataLoadHandle):
    def __init__(self):
        super().__init__()

    def run(self):
        # 计算矩阵
        tt = TfidfVectorizer(max_df=0.5)
        tf = tt.fit_transform(self.train_documents)
        # 非线性SVM模型
        # ovr:一对多策略，ovo表示一对一
        rbf_svm = SVC(kernel='rbf', decision_function_shape='ovo')
        # 模型在训练数据集上的拟合
        rbf_svm.fit(tf, self.train_labels)
        test_tf = TfidfVectorizer(max_df=0.5, vocabulary=tt.vocabulary_)
        test_features = test_tf.fit_transform(self.test_documents)
        predicted = rbf_svm.predict(test_features)
        return self.assess(predicted)


if __name__ == '__main__':
    svn = SVM()
    print(svn.run())
