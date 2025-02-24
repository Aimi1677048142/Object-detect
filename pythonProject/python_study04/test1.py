# 1. 当前目录下，创建一个 txt 文件，写入内容：白发并非雪可替相逢已是上上签
import os.path
import random
import string

# with open('test.txt', mode='w', encoding='utf-8') as file:
#     file.write("白发并非雪可替相逢已是上上签")


# 在上面的文件中，插入标题 《自强人》，再”替“字后面插入换行
# with open('test.txt', mode='r', encoding='utf-8') as file:
#     content = file.read()
#     # print(content.index('替'))
# new_content = "《自强人》\n"+content[:content.index('替')+1]+"\n"+content[content.index('替')+1:]
# with open('test.txt', mode='w', encoding='utf-8') as file1:
#     file1.write(new_content)


# 2. 创建一个目录 file_txt，生成 100 个 txt文件，且每个 txt 文件随机写入 5 个 a~z 组成的字符串，
# 文件名命令规则 001.txt ~ 100.txt
# for i in range(3):
#     file_name = f'{i+1:03}.txt'
#     with open(file_name,mode='w',encoding='utf-8') as file:
#         file.write(''.join(random.choices(string.ascii_lowercase,k=5)))

# 从上面生成的 txt 文件内容中，读取含有 a 这个字符串的文件路径名
# for i in range(3):
#     file_name = f'{i + 1:03}.txt'
#     with open(file_name , mode='r',encoding='utf-8') as file:
#         content = file.read()
#         if 'a' in content:
#             abspath = os.path.abspath(file_name)
#             print(abspath)

# 3. 编写一个程序，从一个文本文件中读取每一行，并将每行的内容逆序写入另一个文件中
with open('test.txt', mode='r', encoding='utf-8') as file, open('test1.txt', mode='w', encoding='utf-8') as file1:
    splitlines = file.read().splitlines()
    print(splitlines)

    file1.writelines(x[::-1] + '\n' for x in splitlines)
