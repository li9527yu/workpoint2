import pickle

def load_pkl(file_path):
    """
    读取 .pkl 文件并返回其内容。

    参数:
        file_path (str): .pkl 文件的路径。

    返回:
        反序列化后的 Python 对象。
    """
    try:
        with open(file_path, 'rb') as f:  # 以二进制模式打开文件
            data = pickle.load(f)  # 反序列化
        return data
    except Exception as e:
        print(f"读取 .pkl 文件时出错: {e}")
        return None

# 示例用法
if __name__ == "__main__":
    # 假设有一个 .pkl 文件路径
    pkl_file = "/data/lzy1211/code/A2II/instructBLIP/img_data/twitter2015/test.pkl"
    
    # 读取 .pkl 文件
    data = load_pkl(pkl_file)
    
    if data is not None:
        print("读取成功！")
        print("文件内容:", data)