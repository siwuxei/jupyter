#导入包
from sklearn.feature_extraction import DictVectorizer

def dictvec():
    # 实例化
    dict = DictVectorizer()
    # 调用fit_transform
    data = dict.fit_transform([{'city': '北京', 'temperature': 100},{'city': '上海', 'temperature': 60},{'city': '深圳', 'temperature': 30}])
    print(data)
    return None

if __name__ == "__main__":
    dictvec()