import tqdm


def main():
    train_data_file = open('./data/train.csv', 'a+')
    public_test_file = open('./data/public_test.csv', 'a+')
    private_test_file = open('./data/private_test.csv', 'a+')
    with open('./data/FER2013.csv', 'r') as file:
        for line in tqdm.tqdm(file.readlines()[1:]):
            line = line.strip().split(',')
            if line[1] == 'Training':
                train_data_file.write(','.join([line[0], line[2]]) + '\n')
            elif line[1] == 'PublicTest':
                public_test_file.write(','.join([line[0], line[2]]) + '\n')
            elif line[1] == 'PrivateTest':
                private_test_file.write(','.join([line[0], line[2]]) + '\n')
    train_data_file.close()
    public_test_file.close()
    private_test_file.close()


if __name__ == '__main__':
    main()
