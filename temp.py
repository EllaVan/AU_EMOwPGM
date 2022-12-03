import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # 在参数构造器中添加两个命令行参数
    parser.add_argument('--name', type=str, default='Siri')
    parser.add_argument('--message', type=str, default=',Welcom to Python World!')

    # 获取所有的命令行参数
    args = parser.parse_args()

    return args


def temp_main(args):
    print('Hi ' + str(args.name) + str(args.message))

if __name__=='__main__':
    args = get_args()
    temp_main(args)