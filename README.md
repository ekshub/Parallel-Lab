# 进行编译前需要修改main.cpp中的头文件引用和宏Parallel_Level为对应的并行度，但注意不要同时包含多个头文件
# 数据文件太大，无法直接上传。请把待加密的数据集命名为‘guesses.txt’放到md5文件夹下，换行来分割口令。
# g++ main.cpp md5_2x.cpp -o main 编译二路并行程序
# g++ main.cpp md5_4x.cpp -o main 编译四路并行程序
# g++ -mavx2 main.cpp md5_8x.cpp -o main 编译八路并行程序
# g++ main.cpp md5_o.cpp -o main 或 g++ test.cpp md5_o.cpp -o test 编译串行程序,后者无需修改任何参数。
