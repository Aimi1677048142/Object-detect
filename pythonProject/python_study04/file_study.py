# open打开文件，返回文件对象
# open('文件路径', mode='文件模式', encoding='utf-8|gbk')
# mode 文件模式：r读，w写（覆盖），x写（文件存在，报错，创建文件），a追加，b二进制，t文本，+读写
f = open('test.txt', mode='r', encoding='utf-8')
f1 = open('test1.txt', mode='a', encoding='utf-8')
f1.write("你好啊，奥特曼")
f1.close()
