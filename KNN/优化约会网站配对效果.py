import numpy as np

def file(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberlines = len(lines)
    Mat = np.zeros((numberlines,3))
    classLabelVector=[]  #定义一个列表
    index = 0  #行的索引
    for line in lines:
        line = line.strip() 
        listFromLine = line.split('\t') #line.split()字符串内按照空格进行分割
        Mat[index,:]= listFromLine[0:3]  #将数据前三列提取出来，index作为每一行的索引
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1]=='largeDoses':
            classLabelVector.append(3)
        index+=1
    return Mat, classLabelVector

if __name__ == '__main__':
    filename = "data.txt"
    datingDataMat, datingLabels = file(filename)
    print(datingDataMat)
    print(datingLabels)
